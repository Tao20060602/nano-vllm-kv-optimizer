"""M12-MVP sparse layer runtime: GPU representatives + GPU selector + CPU block gather.

One M12LayerRuntime per attention layer. The full post-RoPE history lives on
CPU in block-major layout [num_blocks, block_size, Hkv, D]. Representatives are
built on GPU from the final post-RoPE K and stay GPU-resident.  During decode
the selector runs entirely on GPU, only block IDs cross to CPU, selected blocks
are batch-gathered into a reusable pinned staging, and H2D feeds a pre-allocated
packed GPU buffer that also holds sink and recent K/V.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import torch
import torch.nn.functional as F


@dataclass
class M12Config:
    block_size: int = 64
    r: int = 4
    recent_tokens: int = 512
    sink_tokens: int = 64
    top_k_blocks: int = 32
    max_model_len: int = 131072
    num_heads: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    dtype: torch.dtype = torch.bfloat16
    scale: float = 128.0 ** -0.5


class M12LayerRuntime:
    """Per-layer GPU-representative sparse runtime."""

    def __init__(self, layer_id: int, cfg: M12Config):
        self.layer_id = layer_id
        self.cfg = cfg
        self.enabled = True

        B = cfg.block_size
        Hkv = cfg.num_kv_heads
        D = cfg.head_dim
        dtype = cfg.dtype

        self.max_blocks = (cfg.max_model_len + B - 1) // B
        self.groups_per_kv = cfg.num_heads // cfg.num_kv_heads

        # ---- CPU KV: block-major, pageable (not pinned), LAZY-allocated
        # at prefill to the actual sequence length (avoid reserving the full
        # 128K footprint at construction, which would OOM small host machines).
        self.k_cpu = None
        self.v_cpu = None
        self.cpu_blocks_cap = 0
        self.nblocks_filled = 0

        # ---- GPU representatives: [Hkv, max_blocks, r, D] BF16 --------
        self.reps_gpu = torch.empty(
            Hkv, self.max_blocks, cfg.r, D, dtype=dtype, device="cuda"
        )

        # ---- GPU sink and recent windows -------------------------------
        self.sink_k = torch.zeros(cfg.sink_tokens, Hkv, D, dtype=dtype, device="cuda")
        self.sink_v = torch.zeros(cfg.sink_tokens, Hkv, D, dtype=dtype, device="cuda")
        self.recent_k = torch.zeros(cfg.recent_tokens, Hkv, D, dtype=dtype, device="cuda")
        self.recent_v = torch.zeros(cfg.recent_tokens, Hkv, D, dtype=dtype, device="cuda")

        # ---- Pre-allocated packed GPU buffer ---------------------------
        # Layout: [selected_hist | sink | recent]
        self.max_selected_tokens = cfg.top_k_blocks * B
        self.packed_total = (
            self.max_selected_tokens + cfg.sink_tokens + cfg.recent_tokens
        )
        self.packed_k = torch.empty(self.packed_total, Hkv, D, dtype=dtype, device="cuda")
        self.packed_v = torch.empty(self.packed_total, Hkv, D, dtype=dtype, device="cuda")

        # ---- Reusable pinned staging (CPU) ----------------------------
        self.stage_k = torch.empty(
            self.max_selected_tokens, Hkv, D, dtype=dtype, device="cpu", pin_memory=True
        )
        self.stage_v = torch.empty(
            self.max_selected_tokens, Hkv, D, dtype=dtype, device="cpu", pin_memory=True
        )

        # ---- State -----------------------------------------------------
        self.valid_len = 0
        self.prefill_len = 0
        self.last_block_ids: torch.Tensor | None = None
        self.protected_blocks: set[int] = set()
        self.timings: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Representative construction on GPU
    # ------------------------------------------------------------------
    def _build_reps_gpu(self, k_blocks: torch.Tensor, nblocks: int):
        """Batched GPU construction of r real-key representatives.

        k_blocks: [nblocks, block_size, Hkv, D] BF16, post-RoPE real K.
        All blocks and KV heads are processed in parallel (no per-block Python
        loop). mean-direction real key + farthest-point directional coverage;
        normalization is used ONLY for selection, stored reps are raw real K.
        """
        cfg = self.cfg
        B, Hkv, D, r = cfg.block_size, cfg.num_kv_heads, cfg.head_dim, cfg.r
        N = nblocks
        dev = k_blocks.device

        K = k_blocks[:N].to(torch.float32)          # [N, B, H, D]
        K_dir = F.normalize(K, dim=-1)              # [N, B, H, D]
        # Rearrange head before token for gather along the token axis.
        Kt = K.permute(0, 2, 1, 3)                  # [N, H, B, D]
        Kdt = K_dir.permute(0, 2, 1, 3)             # [N, H, B, D]

        # mean direction per (block, kv head)
        mean_dir = F.normalize(K_dir.mean(dim=1), dim=-1)   # [N, H, D]
        cos0 = torch.einsum("nbhd,nhd->nbh", K_dir, mean_dir)  # [N, B, H]
        sel_idx = torch.empty(N, Hkv, r, dtype=torch.long, device=dev)
        sel_idx[:, :, 0] = cos0.argmax(dim=1)        # [N, H]

        INF = 2.0
        for step in range(1, r):
            idx = sel_idx[:, :, :step]               # [N, H, step]
            # gather selected directions: [N, H, step, D]
            sel_dirs = torch.gather(
                Kdt, 2, idx.unsqueeze(-1).expand(N, Hkv, step, D))
            # cosine of every token with all selected reps
            cos_sim = torch.einsum("nbhd,nhsd->nbhs", K_dir, sel_dirs)  # [N,B,H,step]
            max_cos = cos_sim.amax(dim=-1)           # [N, B, H]
            # exclude already-selected tokens (force them high)
            with torch.no_grad():
                mask = torch.zeros(N, B, Hkv, device=dev)
                mask.scatter_(1, idx.permute(0, 2, 1).reshape(N, step, Hkv), INF)
            max_cos = torch.maximum(max_cos, mask)
            # worst covered = minimum max-cosine
            sel_idx[:, :, step] = max_cos.argmin(dim=1)   # [N, H]

        # Gather RAW real K at chosen indices: [N, H, r, D]
        raw = torch.gather(
            Kt, 2, sel_idx.unsqueeze(-1).expand(N, Hkv, r, D))
        # Store as [Hkv, N, r, D] BF16
        self.reps_gpu[:, :N, :, :] = raw.permute(1, 0, 2, 3).to(self.reps_gpu.dtype)

    # ------------------------------------------------------------------
    # Prefill
    # ------------------------------------------------------------------
    def prefill(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> dict:
        """Store post-RoPE prefill K/V to CPU and build GPU reps.

        q,k,v: [T, Hq/Hkv, D] GPU tensors (post-QK-Norm, post-YaRN-RoPE).
        """
        cfg = self.cfg
        B = cfg.block_size
        T, Hkv, D = k.shape
        assert T >= 2

        padded = ((T + B - 1) // B) * B
        if padded > T:
            k_pad = F.pad(k, (0, 0, 0, 0, 0, padded - T))
            v_pad = F.pad(v, (0, 0, 0, 0, 0, padded - T))
        else:
            k_pad, v_pad = k, v

        nblocks = padded // B
        k_blocks = k_pad.view(nblocks, B, Hkv, D)
        v_blocks = v_pad.view(nblocks, B, Hkv, D)

        # Lazily (re)allocate pageable CPU KV to actual block count.
        t_alloc = perf_counter()
        if self.k_cpu is None or self.cpu_blocks_cap < nblocks:
            self.k_cpu = torch.empty(nblocks, B, Hkv, D, dtype=cfg.dtype, device="cpu")
            self.v_cpu = torch.empty(nblocks, B, Hkv, D, dtype=cfg.dtype, device="cpu")
            self.cpu_blocks_cap = nblocks
        alloc_ms = (perf_counter() - t_alloc) * 1000.0

        # Store to CPU (pageable, not pinned)
        t0 = perf_counter()
        self.k_cpu[:nblocks].copy_(k_blocks.cpu())
        self.v_cpu[:nblocks].copy_(v_blocks.cpu())
        self.nblocks_filled = nblocks
        cpu_store_ms = (perf_counter() - t0) * 1000.0 + alloc_ms

        # Build reps on GPU
        t1 = perf_counter()
        self._build_reps_gpu(k_blocks, nblocks)
        reps_ms = (perf_counter() - t1) * 1000.0

        # Sink: first sink_tokens
        sink_len = min(cfg.sink_tokens, T)
        self.sink_k[:sink_len].copy_(k[:sink_len])
        self.sink_v[:sink_len].copy_(v[:sink_len])

        # Recent: last recent_tokens
        recent_len = min(cfg.recent_tokens, T)
        self.recent_k[:recent_len].copy_(k[-recent_len:])
        self.recent_v[:recent_len].copy_(v[-recent_len:])

        self.valid_len = T
        self.prefill_len = T

        # Historical blocks exclude sink (block 0) and recent (last blocks)
        n_recent_blocks = cfg.recent_tokens // B
        sink_blocks = {0}
        recent_blocks = set(range(max(1, nblocks - n_recent_blocks), nblocks))
        self.protected_blocks = sink_blocks | recent_blocks

        return {
            "cpu_store_ms": cpu_store_ms,
            "reps_build_ms": reps_ms,
            "nblocks": nblocks,
        }

    # ------------------------------------------------------------------
    # GPU selector (fully vectorized)
    # ------------------------------------------------------------------
    def _gpu_select(self, q: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """GPU-vectorized GQA group-score + global temporal top-k.

        q: [Hq, D] GPU, post-RoPE decode query.
        Returns (block_ids_cpu [K] int64, timing_dict).
        """
        cfg = self.cfg
        Hq, Hkv, D = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
        nblocks = self.nblocks_filled

        q_g = q.view(Hkv, self.groups_per_kv, D)  # [Hkv, G, D]

        # GQA group score: max over Q heads in group, max over r reps
        scores = torch.einsum(
            "hgd,hbrd->hgbr", q_g.float(), self.reps_gpu[:, :nblocks].float()
        )  # [Hkv, G, nblocks, r]
        scores = scores.amax(dim=-1)        # max over r: [Hkv, G, nblocks]
        scores = scores.amax(dim=1)         # max over Q heads: [Hkv, nblocks]
        global_scores = scores.amax(dim=0)  # global temporal: [nblocks]

        # Mask protected blocks (sink/recent)
        if self.protected_blocks:
            prot = torch.tensor(sorted(self.protected_blocks),
                                dtype=torch.long, device=global_scores.device)
            prot = prot[prot < nblocks]
            if prot.numel() > 0:
                global_scores[prot] = float("-inf")

        n_prot = len([p for p in self.protected_blocks if p < nblocks])
        avail = nblocks - n_prot
        k = min(cfg.top_k_blocks, max(1, avail))
        _, topk_ids = global_scores.topk(k)

        t_d2h = perf_counter()
        block_ids_cpu = topk_ids.cpu()
        d2h_ms = (perf_counter() - t_d2h) * 1000.0

        return block_ids_cpu, {"d2h_ms": d2h_ms, "n_selected": k}

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------
    def decode(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
               device: torch.device) -> torch.Tensor:
        """One decode step. q/k/v: [1, Hq/Hkv, D] GPU post-RoPE."""
        cfg = self.cfg
        B = cfg.block_size
        Hkv = cfg.num_kv_heads
        D = cfg.head_dim
        Hq = cfg.num_heads

        assert q.shape[0] == 1 and k.shape[0] == 1

        # 1. Append current token to recent window (shift left; clone avoids
        #    overlapping-copy error, buffer itself is reused)
        t0 = perf_counter()
        self.recent_k[:-1].copy_(self.recent_k[1:].clone())
        self.recent_v[:-1].copy_(self.recent_v[1:].clone())
        self.recent_k[-1].copy_(k[0])
        self.recent_v[-1].copy_(v[0])
        recent_ms = (perf_counter() - t0) * 1000.0

        # 2. GPU selector -> block IDs D2H
        t_sel = perf_counter()
        block_ids_cpu, sel_info = self._gpu_select(q[0])
        selector_ms = (perf_counter() - t_sel) * 1000.0

        K = block_ids_cpu.shape[0]
        sel_hist_tokens = K * B

        # 3. CPU batch gather into pinned staging (vectorized index, no per-block loop)
        t_g = perf_counter()
        sel_k = self.k_cpu[block_ids_cpu].reshape(sel_hist_tokens, Hkv, D)
        sel_v = self.v_cpu[block_ids_cpu].reshape(sel_hist_tokens, Hkv, D)
        self.stage_k[:sel_hist_tokens].copy_(sel_k)
        self.stage_v[:sel_hist_tokens].copy_(sel_v)
        gather_ms = (perf_counter() - t_g) * 1000.0

        # 4. Assemble packed GPU buffer: [selected_hist | sink | recent]
        t_h = perf_counter()
        off = 0
        self.packed_k[off:off + sel_hist_tokens].copy_(
            self.stage_k[:sel_hist_tokens], non_blocking=True)
        self.packed_v[off:off + sel_hist_tokens].copy_(
            self.stage_v[:sel_hist_tokens], non_blocking=True)
        off += sel_hist_tokens

        sink_len = min(cfg.sink_tokens, self.valid_len)
        self.packed_k[off:off + sink_len].copy_(self.sink_k[:sink_len])
        self.packed_v[off:off + sink_len].copy_(self.sink_v[:sink_len])
        off += sink_len

        recent_len = min(cfg.recent_tokens, self.valid_len + 1)
        self.packed_k[off:off + recent_len].copy_(self.recent_k[-recent_len:])
        self.packed_v[off:off + recent_len].copy_(self.recent_v[-recent_len:])
        off += recent_len
        total_attend = off
        h2d_ms = (perf_counter() - t_h) * 1000.0

        # 5. Packed exact attention (real selected K/V, not reps)
        t_a = perf_counter()
        q_f = q[0].float()  # [Hq, D]
        k_f = self.packed_k[:total_attend].float()  # [S, Hkv, D]
        v_f = self.packed_v[:total_attend].float()

        q_per_kv = Hq // Hkv
        q_exp = q_f.view(Hkv, q_per_kv, D)  # [Hkv, G, D]
        scores = torch.einsum("hgd,shd->hgs", q_exp, k_f)  # [Hkv, G, S]
        scores = scores * cfg.scale
        attn = F.softmax(scores, dim=-1)  # [Hkv, G, S]
        out = torch.einsum("hgs,shd->hgd", attn, v_f)  # [Hkv, G, D]
        out = out.reshape(Hq, D).to(q.dtype)
        attn_ms = (perf_counter() - t_a) * 1000.0

        self.valid_len += 1
        self.last_block_ids = block_ids_cpu

        self.timings = {
            "selector_ms": selector_ms,
            "d2h_ms": sel_info.get("d2h_ms", 0.0),
            "cpu_gather_ms": gather_ms,
            "h2d_pack_ms": h2d_ms,
            "recent_ms": recent_ms,
            "packed_attn_ms": attn_ms,
            "selected_tokens": sel_hist_tokens,
            "total_attend_tokens": total_attend,
        }

        if not bool(torch.isfinite(out).all()):
            raise RuntimeError("M12 sparse decode output is non-finite")

        return out.unsqueeze(0)
