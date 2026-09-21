"""M12-MVP sparse layer runtime: GPU representatives + GPU selector + CPU block gather.

One M12LayerRuntime per attention layer. The full post-RoPE history lives on
CPU in block-major layout [num_blocks, block_size, Hkv, D]. Representatives are
built on GPU from the final post-RoPE K and stay GPU-resident.  During decode
the selector runs entirely on GPU, only block IDs cross to CPU, selected blocks
are batch-gathered into a reusable pinned staging, and H2D feeds a pre-allocated
packed GPU buffer that also holds sink and recent K/V.

Chunked prefill: the 128K prompt is prefilled in chunks. The first chunk uses a
dense within-chunk FlashAttention (no history yet); every later chunk runs a
single fused exact attention over [sink | selected historical blocks | current
chunk (causal)] so later context genuinely attends earlier context in ONE
softmax -- never independent per-chunk computation stitched together.
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

        # ---- CPU KV: block-major, pageable (not pinned), LAZY-allocated ----
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
    # Representative construction on GPU (batched over blocks and heads)
    # ------------------------------------------------------------------
    def _build_reps_gpu(self, k_blocks: torch.Tensor, start: int, nblocks: int):
        """Build r real-key representatives on GPU per (block, kv_head).

        k_blocks: [nblocks, block_size, Hkv, D] BF16, post-RoPE real K.
        Writes self.reps_gpu[:, start:start+nblocks, :, :].
        mean-direction real key + farthest-point directional coverage;
        normalization is used ONLY for selection; stored reps are raw real K.
        """
        cfg = self.cfg
        B, Hkv, D, r = cfg.block_size, cfg.num_kv_heads, cfg.head_dim, cfg.r
        N = nblocks
        dev = k_blocks.device

        K = k_blocks[:N].to(torch.float32)          # [N, B, H, D]
        K_dir = F.normalize(K, dim=-1)              # [N, B, H, D]
        Kt = K.permute(0, 2, 1, 3)                  # [N, H, B, D]
        Kdt = K_dir.permute(0, 2, 1, 3)             # [N, H, B, D]

        mean_dir = F.normalize(K_dir.mean(dim=1), dim=-1)   # [N, H, D]
        cos0 = torch.einsum("nbhd,nhd->nbh", K_dir, mean_dir)  # [N, B, H]
        sel_idx = torch.empty(N, Hkv, r, dtype=torch.long, device=dev)
        sel_idx[:, :, 0] = cos0.argmax(dim=1)        # [N, H]

        INF = 2.0
        for step in range(1, r):
            idx = sel_idx[:, :, :step]               # [N, H, step]
            sel_dirs = torch.gather(
                Kdt, 2, idx.unsqueeze(-1).expand(N, Hkv, step, D))
            cos_sim = torch.einsum("nbhd,nhsd->nbhs", K_dir, sel_dirs)  # [N,B,H,step]
            max_cos = cos_sim.amax(dim=-1)           # [N, B, H]
            with torch.no_grad():
                mask = torch.zeros(N, B, Hkv, device=dev)
                mask.scatter_(1, idx.permute(0, 2, 1).reshape(N, step, Hkv), INF)
            max_cos = torch.maximum(max_cos, mask)
            sel_idx[:, :, step] = max_cos.argmin(dim=1)   # [N, H]

        raw = torch.gather(
            Kt, 2, sel_idx.unsqueeze(-1).expand(N, Hkv, r, D))
        self.reps_gpu[:, start:start + N, :, :] = (
            raw.permute(1, 0, 2, 3).to(self.reps_gpu.dtype))

    # ------------------------------------------------------------------
    # Store K/V to CPU + build reps + refresh sink/recent/protected
    # ------------------------------------------------------------------
    def _store_kv(self, k: torch.Tensor, v: torch.Tensor):
        """Append a prefill chunk's post-RoPE K/V to the CPU history.

        k,v: [T, Hkv, D] GPU tensors.
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

        start_block = self.nblocks_filled
        nblocks = padded // B
        k_blocks = k_pad.view(nblocks, B, Hkv, D)
        v_blocks = v_pad.view(nblocks, B, Hkv, D)

        # Lazily grow pageable CPU KV (never pinned).
        need = start_block + nblocks
        if self.k_cpu is None or self.cpu_blocks_cap < need:
            new_cap = max(need, 2 * (self.cpu_blocks_cap or 1))
            new_k = torch.empty(new_cap, B, Hkv, D, dtype=cfg.dtype, device="cpu")
            new_v = torch.empty(new_cap, B, Hkv, D, dtype=cfg.dtype, device="cpu")
            if self.k_cpu is not None and start_block > 0:
                new_k[:start_block].copy_(self.k_cpu[:start_block])
                new_v[:start_block].copy_(self.v_cpu[:start_block])
            self.k_cpu, self.v_cpu = new_k, new_v
            self.cpu_blocks_cap = new_cap

        self.k_cpu[start_block:need].copy_(k_blocks.cpu())
        self.v_cpu[start_block:need].copy_(v_blocks.cpu())

        # Build reps for the new blocks on GPU.
        self._build_reps_gpu(k_blocks, start_block, nblocks)

        # Sink: earliest tokens (first chunk only).
        if self.prefill_len == 0:
            sink_len = min(cfg.sink_tokens, T)
            self.sink_k[:sink_len].copy_(k[:sink_len])
            self.sink_v[:sink_len].copy_(v[:sink_len])

        # Recent: tail of everything seen so far.
        recent_len = min(cfg.recent_tokens, self.valid_len + T)
        if T >= recent_len:
            self.recent_k[:recent_len].copy_(k[-recent_len:])
            self.recent_v[:recent_len].copy_(v[-recent_len:])
        else:
            carry = recent_len - T
            self.recent_k[:carry].copy_(self.recent_k[recent_len - carry:recent_len])
            self.recent_v[:carry].copy_(self.recent_v[recent_len - carry:recent_len])
            self.recent_k[carry:recent_len].copy_(k)
            self.recent_v[carry:recent_len].copy_(v)

        self.valid_len += T
        self.prefill_len += T
        self.nblocks_filled = need

        # Historical blocks exclude sink (block 0) and recent (tail blocks).
        n_recent_blocks = cfg.recent_tokens // B
        sink_blocks = {0}
        recent_blocks = set(range(max(1, need - n_recent_blocks), need))
        self.protected_blocks = sink_blocks | recent_blocks

    # ------------------------------------------------------------------
    # First prefill chunk: dense within-chunk attention (no history yet)
    # ------------------------------------------------------------------
    def prefill_first(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> dict:
        """Store the first chunk; dense attention is done by the caller (FA2)."""
        assert self.prefill_len == 0, "prefill_first called after history exists"
        self._store_kv(k, v)
        return {"nblocks": self.nblocks_filled, "valid_len": self.valid_len}

    # ------------------------------------------------------------------
    # Later prefill chunk: fused exact attention over sink+history+chunk
    # ------------------------------------------------------------------
    def prefill_chunk(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                      device: torch.device) -> torch.Tensor:
        """Fused exact attention for a later prefill chunk.

        q,k,v: [T, Hq/Hkv, D] GPU, post-RoPE.
        Attention attends [sink(64) | top-k historical blocks | current chunk
        (causal)] in ONE softmax (never independent per-chunk stitching).
        """
        cfg = self.cfg
        B = cfg.block_size
        T, Hkv, D = k.shape
        Hq = cfg.num_heads
        assert self.prefill_len > 0, "prefill_chunk requires prior history"

        # 1. Select historical blocks for this chunk (mean query direction).
        q_mean = q.mean(dim=0)  # [Hq, D]
        hist_ids, sel_info = self._gpu_select(q_mean)  # [K] CPU

        # 2. CPU batch gather selected historical blocks into pinned staging.
        Ksel = hist_ids.shape[0]
        sel_hist_tokens = Ksel * B
        sel_k = self.k_cpu[hist_ids].reshape(sel_hist_tokens, Hkv, D)
        sel_v = self.v_cpu[hist_ids].reshape(sel_hist_tokens, Hkv, D)
        self.stage_k[:sel_hist_tokens].copy_(sel_k)
        self.stage_v[:sel_hist_tokens].copy_(sel_v)

        # 3. Store current chunk into history (so later chunks see it too).
        self._store_kv(k, v)

        # 4. Fused packed exact attention: [hist | sink | chunk(causal)].
        sink_len = cfg.sink_tokens
        total = sel_hist_tokens + sink_len + T
        k_pack = torch.empty(total, Hkv, D, dtype=k.dtype, device=device)
        v_pack = torch.empty(total, Hkv, D, dtype=k.dtype, device=device)

        off = 0
        k_pack[off:off + sel_hist_tokens] = self.stage_k[:sel_hist_tokens].to(device)
        v_pack[off:off + sel_hist_tokens] = self.stage_v[:sel_hist_tokens].to(device)
        off += sel_hist_tokens
        k_pack[off:off + sink_len] = self.sink_k[:sink_len]
        v_pack[off:off + sink_len] = self.sink_v[:sink_len]
        off += sink_len
        k_pack[off:off + T] = k
        v_pack[off:off + T] = v

        o = self._fused_attention(
            q, k_pack, v_pack,
            hist_len=sel_hist_tokens, sink_len=sink_len, causal_from=off,
        )
        self.last_block_ids = hist_ids
        return o

    # ------------------------------------------------------------------
    # Fused exact attention (decode or prefill-chunk), per-KV-head loop to
    # keep peak GPU memory bounded.
    # ------------------------------------------------------------------
    def _fused_attention(
        self,
        q: torch.Tensor,          # [Tq, Hq, D] GPU
        k: torch.Tensor,          # [S, Hkv, D] GPU
        v: torch.Tensor,          # [S, Hkv, D] GPU
        hist_len: int,
        sink_len: int,
        causal_from: int,         # start index of the causal (current-chunk) region
    ) -> torch.Tensor:
        cfg = self.cfg
        Tq = q.shape[0]
        S = k.shape[0]
        Hq = cfg.num_heads
        Hkv = cfg.num_kv_heads
        D = cfg.head_dim
        G = Hq // Hkv

        q_f = q.float()                     # [Tq, Hq, D]
        k_f = k.float()                     # [S, Hkv, D]
        v_f = v.float()
        qg = q_f.view(Tq, Hkv, G, D)        # [Tq, Hkv, G, D]

        # causal mask for the chunk region: per-query-row [Tq, S]
        # query t (chunk-relative position t) sees history + chunk keys <= t
        mask = torch.zeros(Tq, S, dtype=torch.bool, device=q.device)
        if causal_from < S:
            for t in range(Tq):
                mask[t, causal_from + t + 1:] = True

        out = torch.empty(Tq, Hq, D, dtype=q_f.dtype, device=q.device)
        for h in range(Hkv):
            qh = qg[:, h]                   # [Tq, G, D]
            kh = k_f[:, h, :]               # [S, D]
            vh = v_f[:, h, :]               # [S, D]
            # scores: [Tq, G, S]
            scores = torch.einsum("tgd,sd->tgs", qh, kh) * cfg.scale
            scores = scores.masked_fill(mask.unsqueeze(1), float("-inf"))  # [Tq,G,S]
            attn = F.softmax(scores, dim=-1)
            o = torch.einsum("tgs,sd->tgd", attn, vh)   # [Tq, G, D]
            out[:, h * G:(h + 1) * G, :] = o
        return out.to(q.dtype)

    # ------------------------------------------------------------------
    # GPU selector (fully vectorized)
    # ------------------------------------------------------------------
    def _gpu_select(self, q: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """GPU-vectorized GQA group-score + global temporal top-k.

        q: [Hq, D] GPU, post-RoPE query (single token or mean direction).
        Returns (block_ids_cpu [K] int64, timing_dict).
        """
        cfg = self.cfg
        Hq, Hkv, D = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
        nblocks = self.nblocks_filled

        q_g = q.view(Hkv, self.groups_per_kv, D)  # [Hkv, G, D]

        scores = torch.einsum(
            "hgd,hbrd->hgbr", q_g.float(), self.reps_gpu[:, :nblocks].float()
        )  # [Hkv, G, nblocks, r]
        scores = scores.amax(dim=-1)        # max over r: [Hkv, G, nblocks]
        scores = scores.amax(dim=1)         # max over Q heads: [Hkv, nblocks]
        global_scores = scores.amax(dim=0)  # global temporal: [nblocks]

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

        # 5. Packed exact attention over [hist | sink | recent]
        t_a = perf_counter()
        o = self._fused_attention(
            q, self.packed_k[:total_attend], self.packed_v[:total_attend],
            hist_len=sel_hist_tokens, sink_len=sink_len, causal_from=total_attend,
        )
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

        if not bool(torch.isfinite(o).all()):
            raise RuntimeError("M12 sparse decode output is non-finite")

        return o
