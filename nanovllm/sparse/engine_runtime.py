"""Per-layer appendable CPU K/V history + selector dispatch for M11.

One :class:`SparseLayerRuntime` is owned by each shared ``Attention`` module when
sparse mode is enabled.  The full post-RoPE history of every layer lives on CPU
in the original model dtype ``[num_tokens, Hkv, D]``.  A decode step appends the
current token, selects historical blocks with the configured selector, gathers
only the packed selection to a pinned staging buffer, H2D-copies it and runs a
packed PyTorch attention on GPU.  No paged GPU KV cache is touched.

This is an educational single-sequence prototype: no online graph insertion,
no asynchronous overlap, no production batching.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import torch

from nanovllm.sparse.block_sparse import (
    exact_block_scores,
    gqa_token_scores,
    kv_head_for_query,
    select_dipr_blocks,
    select_topk_blocks,
    selected_token_indices,
    sparse_decode_attention,
    union_block_mask,
)
from nanovllm.sparse.representatives import (
    BlockRepresentatives,
    flat_mean_select,
    flat_real_select,
)
from nanovllm.sparse.graph_diprs import BlockGraphIndex


# -- shared invocation counters (single active request) -------------------
@dataclass
class SparseCounters:
    sparse_prefill_initializations: int = 0
    sparse_decode_layer_calls: int = 0
    sparse_generated_steps: int = 0
    dense_decode_fallbacks: int = 0

    def reset(self) -> None:
        self.sparse_prefill_initializations = 0
        self.sparse_decode_layer_calls = 0
        self.sparse_generated_steps = 0
        self.dense_decode_fallbacks = 0


_COUNTERS = SparseCounters()
_LAYER_RUNTIMES: list["SparseLayerRuntime"] = []


def get_sparse_counters() -> SparseCounters:
    return _COUNTERS


def reset_sparse_counters() -> None:
    _COUNTERS.reset()


def reset_all_layer_runtimes() -> None:
    for rt in _LAYER_RUNTIMES:
        rt.reset()


def set_all_selectors(name: str) -> None:
    for rt in _LAYER_RUNTIMES:
        rt.selector = name


@dataclass
class SparseEngineConfig:
    selector: str
    rbs: int
    recent_tokens: int
    first_tokens: int
    top_k: int
    beta_raw: float
    num_representatives: int
    graph_degree: int
    graph_l0: int
    graph_max_scored: int
    graph_projection_topk: int
    query_samples: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    dtype: torch.dtype
    scale: float
    max_model_len: int


@dataclass
class DecodeSelection:
    """Common result returned by every selector for one decode step/layer."""
    union_mask: torch.Tensor          # bool [B] CPU, pre-window (frozen-prefix blocks)
    num_selected_tokens: int
    selected_token_ratio: float
    work: dict = field(default_factory=dict)
    search_ms: float = 0.0
    refine_ms: float = 0.0
    gather_ms: float = 0.0
    h2d_ms: float = 0.0
    attention_ms: float = 0.0
    h2d_bytes: int = 0
    full_kv_bytes: int = 0
    truncated: bool = False
    selector: str = ""
    layer_id: int = -1
    step: int = -1


def _sample_queries(q: torch.Tensor, n: int) -> torch.Tensor:
    """Deterministically sample non-final prefill query positions."""
    T = q.shape[0]
    if T <= 1:
        raise ValueError("prefill must have at least one non-final query position")
    n_choose = min(n, T - 1)
    if n_choose <= 0:
        raise ValueError("no non-final query positions to sample")
    # deterministic, spread across [0, T-2]
    if n_choose == T - 1:
        idx = torch.arange(T - 1)
    else:
        idx = torch.linspace(0, T - 2, steps=n_choose).round().long()
    return q[idx]


class SparseLayerRuntime:
    """One layer's CPU-resident K/V history + selector state."""

    def __init__(self, layer_id: int, cfg: SparseEngineConfig):
        self.layer_id = layer_id
        self.cfg = cfg
        self.enabled = True
        Hkv, D = cfg.num_kv_heads, cfg.head_dim
        dtype = cfg.dtype
        self.capacity = int(cfg.max_model_len)
        # pageable full history (preferred over pinning every layer)
        self.k_cpu = torch.empty(self.capacity, Hkv, D, dtype=dtype, device="cpu")
        self.v_cpu = torch.empty(self.capacity, Hkv, D, dtype=dtype, device="cpu")
        self.valid_len = 0
        self.prefill_len = 0
        self.step = 0
        # frozen index/representative state (built once at prefill)
        self.reps: BlockRepresentatives | None = None
        self.graph: BlockGraphIndex | None = None
        self.k_frozen_f32: torch.Tensor | None = None
        self.q_samples: torch.Tensor | None = None
        self.selector = cfg.selector
        # last selection record for diagnostics
        self.last_selection: DecodeSelection | None = None
        _LAYER_RUNTIMES.append(self)

    # -- lifecycle -------------------------------------------------------
    def reset(self) -> None:
        self.valid_len = 0
        self.prefill_len = 0
        self.step = 0
        self.reps = None
        self.graph = None
        self.k_frozen_f32 = None
        self.q_samples = None
        self.last_selection = None

    @property
    def num_blocks(self) -> int:
        return (self.prefill_len + self.cfg.rbs - 1) // self.cfg.rbs

    def history_bytes(self) -> int:
        cap = self.k_cpu.shape[0]
        Hkv, D = self.k_cpu.shape[1], self.k_cpu.shape[2]
        return 2 * cap * Hkv * D * self.k_cpu.element_size()

    # -- prefill ---------------------------------------------------------
    def prefill(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> dict:
        """Store post-RoPE prefill K/V on CPU and build the frozen index."""
        T, Hkv, D = k.shape
        assert T >= 2, "sparse prefill needs >= 2 tokens"
        if T > self.capacity:
            raise ValueError(
                f"sparse prefill prompt ({T}) exceeds capacity {self.capacity}")
        tc = perf_counter()
        self.k_cpu[:T].copy_(k.detach().cpu())
        self.v_cpu[:T].copy_(v.detach().cpu())
        cpu_copy_ms = (perf_counter() - tc) * 1000.0
        self.valid_len = T
        self.prefill_len = T
        self.step = 0
        info: dict[str, float] = {"cpu_copy_ms": cpu_copy_ms}

        t0 = perf_counter()
        k_f32 = self.k_cpu[:T].float()
        self.k_frozen_f32 = k_f32
        self.q_samples = _sample_queries(q.detach().cpu(), self.cfg.query_samples)
        reps = BlockRepresentatives(k_f32, self.cfg.rbs, self.cfg.num_representatives)
        self.reps = reps
        if self.selector in ("knn_graph", "query_guided"):
            g = BlockGraphIndex(reps)
            g.build_knn(self.cfg.graph_degree)
            if self.selector == "query_guided":
                g.build_query_guided(
                    self.q_samples, self.cfg.num_heads,
                    self.cfg.graph_degree, self.cfg.graph_projection_topk,
                )
            self.graph = g
        info["index_build_ms"] = (perf_counter() - t0) * 1000.0
        info["cpu_copy_ms"] = 0.0  # included above (synchronous D2H + build)
        _COUNTERS.sparse_prefill_initializations += 1
        return info

    # -- decode ----------------------------------------------------------
    def _current_k_f32(self) -> torch.Tensor:
        return self.k_cpu[: self.valid_len].float()

    def _run_selector(self, q_f32: torch.Tensor) -> tuple[torch.Tensor, dict, float, float, bool]:
        """Return (union_mask[B_prefill], work, search_ms, refine_ms, truncated)."""
        cfg = self.cfg
        mode = self.selector
        B = self.num_blocks
        work: dict[str, Any] = {}
        truncated = False
        t_search = perf_counter()
        if mode == "full":
            union = torch.ones(B, dtype=torch.bool)
            search_ms = (perf_counter() - t_search) * 1000.0
            return union, work, search_ms, 0.0, False

        if mode in ("exact_dipr", "top_k"):
            kcur = self._current_k_f32()
            ts = gqa_token_scores(q_f32, kcur)
            bs = exact_block_scores(ts, cfg.rbs)
            if mode == "exact_dipr":
                pm = select_dipr_blocks(bs, cfg.beta_raw)
            else:
                pm = select_topk_blocks(bs, cfg.top_k)
            union = union_block_mask(pm)
            search_ms = (perf_counter() - t_search) * 1000.0
            return union, work, search_ms, 0.0, False

        # representative-based modes (frozen prefix)
        if mode == "mean":
            res = flat_mean_select(q_f32, self.reps, cfg.beta_raw)
            union = res.per_head_mask.any(0)
            work["rep_dot_products"] = res.rep_dot_products
            work["refined_block_head_pairs"] = res.refined_block_count
            search_ms, refine_ms = res.rep_scan_ms, res.refine_ms
            return union, work, search_ms, refine_ms, False

        if mode == "real":
            res = flat_real_select(q_f32, self.reps, cfg.beta_raw)
            union = res.per_head_mask.any(0)
            work["rep_dot_products"] = res.rep_dot_products
            work["refined_block_head_pairs"] = res.refined_block_count
            search_ms, refine_ms = res.rep_scan_ms, res.refine_ms
            return union, work, search_ms, refine_ms, False

        if mode in ("knn_graph", "query_guided"):
            Hq = q_f32.shape[0]
            g_of = kv_head_for_query(Hq, cfg.num_kv_heads)
            per = []
            scored_pairs = refined_pairs = 0
            for h in range(Hq):
                g = int(g_of[h].item())
                sr = self.graph.search(
                    q_f32[h], g, cfg.beta_raw, cfg.graph_l0, cfg.graph_max_scored)
                fm = torch.zeros(B, dtype=torch.bool)
                fm[sr.final_blocks] = True
                per.append(fm)
                scored_pairs += len(sr.scored_blocks)
                refined_pairs += len(sr.rep_candidate_blocks)
                truncated = truncated or sr.truncated
            union = torch.stack(per, 0).any(0)
            work["scored_block_head_pairs"] = scored_pairs
            work["refined_block_head_pairs"] = refined_pairs
            search_ms = (perf_counter() - t_search) * 1000.0
            return union, work, search_ms, 0.0, truncated

        raise ValueError(f"unknown selector {mode!r}")

    def decode(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
               device: torch.device) -> torch.Tensor:
        """One decode layer: append current K/V, select, gather, H2D, attend.

        q/k/v are GPU post-RoPE tensors of shape ``[1,Hq,D]/[1,Hkv,D]/[1,Hkv,D]``.
        Returns the packed-attention output ``[1,Hq,D]`` on GPU.
        """
        assert q.shape[0] == 1 and k.shape[0] == 1 and v.shape[0] == 1
        if self.valid_len == 0:
            raise RuntimeError("sparse decode before a complete prefill")
        cfg = self.cfg
        Hq, D = q.shape[1], q.shape[2]
        Hkv = k.shape[1]

        # 1-2. append current post-RoPE K/V to CPU history
        if self.valid_len + 1 > self.capacity:
            raise IndexError(
                f"sparse decode exceeds capacity {self.capacity} "
                f"(valid_len={self.valid_len}); prompt+max_tokens must fit max_model_len")
        self.k_cpu[self.valid_len].copy_(k[0].detach().cpu())
        self.v_cpu[self.valid_len].copy_(v[0].detach().cpu())
        self.valid_len += 1
        self.step += 1
        if self.layer_id == 0:
            _COUNTERS.sparse_generated_steps += 1
        _COUNTERS.sparse_decode_layer_calls += 1

        # 3. select historical blocks (pre-window), query = current q
        q_cpu = q[0].detach().cpu().float()      # [Hq,D]
        t0 = perf_counter()
        union, work, search_ms, refine_ms, truncated = self._run_selector(q_cpu)

        # pre-window retrieval recall vs exact Block-DIPR oracle (diagnostics)
        try:
            from nanovllm.sparse.block_sparse import exact_block_scores, select_dipr_blocks
            kcur = self._current_k_f32()
            escores = exact_block_scores(gqa_token_scores(q_cpu, kcur), cfg.rbs)
            oracle_union = select_dipr_blocks(escores, cfg.beta_raw).any(0)
            L = min(union.shape[0], oracle_union.shape[0])
            denom = int(oracle_union[:L].sum())
            work["pre_window_recall"] = (
                float((union[:L] & oracle_union[:L]).sum()) / denom) if denom else 1.0
            work["oracle_blocks"] = denom
        except Exception:
            pass

        # 4. assemble token indices: pad frozen mask to current block count,
        #    then force first + recent windows (current token always included).
        cur_blocks = (self.valid_len + cfg.rbs - 1) // cfg.rbs
        if union.shape[0] < cur_blocks:
            pad = torch.zeros(cur_blocks - union.shape[0], dtype=torch.bool)
            union = torch.cat([union, pad], 0)
        elif union.shape[0] > cur_blocks:
            union = union[:cur_blocks]
        indices = selected_token_indices(
            union, self.valid_len, cfg.rbs,
            first_tokens=cfg.first_tokens, recent_tokens=cfg.recent_tokens,
        )
        # current token must always be attended
        assert int(indices[-1].item()) == self.valid_len - 1
        assert int(indices[0].item()) >= 0

        # 5. gather sorted unique rows into contiguous pinned staging
        t_g = perf_counter()
        S = int(indices.numel())
        shape = (S, Hkv, D)
        pk = torch.empty(shape, dtype=cfg.dtype, device="cpu", pin_memory=True)
        pv = torch.empty(shape, dtype=cfg.dtype, device="cpu", pin_memory=True)
        torch.index_select(self.k_cpu[: self.valid_len], 0, indices, out=pk)
        torch.index_select(self.v_cpu[: self.valid_len], 0, indices, out=pv)
        gather_ms = (perf_counter() - t_g) * 1000.0

        # 6. H2D: only packed selected K/V (except full selector, labelled)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_h = perf_counter()
        pk_gpu = pk.to(device, non_blocking=False)
        pv_gpu = pv.to(device, non_blocking=False)
        if device.type == "cuda":
            torch.cuda.synchronize()
        h2d_ms = (perf_counter() - t_h) * 1000.0

        # 7. packed PyTorch attention on GPU (no flash_attn_with_kvcache)
        q_gpu = q[0].to(device, non_blocking=False).to(pk_gpu.dtype)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_a = perf_counter()
        packed_pos = torch.arange(S, device=device)
        out = sparse_decode_attention(q_gpu, pk_gpu, pv_gpu, packed_pos, scale=cfg.scale)
        if device.type == "cuda":
            torch.cuda.synchronize()
        attn_ms = (perf_counter() - t_a) * 1000.0

        h2d_bytes = 2 * S * Hkv * D * cfg.dtype.itemsize
        full_bytes = 2 * self.valid_len * Hkv * D * cfg.dtype.itemsize
        if self.selector != "full" and h2d_bytes != int(2 * S * Hkv * D * cfg.dtype.itemsize):
            raise RuntimeError("H2D accounting mismatch")

        sel = DecodeSelection(
            union_mask=union,
            num_selected_tokens=S,
            selected_token_ratio=S / self.valid_len,
            work=work,
            search_ms=search_ms,
            refine_ms=refine_ms,
            gather_ms=gather_ms,
            h2d_ms=h2d_ms,
            attention_ms=attn_ms,
            h2d_bytes=h2d_bytes,
            full_kv_bytes=full_bytes,
            truncated=truncated,
            selector=self.selector,
            layer_id=self.layer_id,
            step=self.step,
        )
        self.last_selection = sel
        if not bool(torch.isfinite(out).all()):
            raise RuntimeError("sparse decode attention output is non-finite")
        return out.unsqueeze(0).to(q.dtype)
