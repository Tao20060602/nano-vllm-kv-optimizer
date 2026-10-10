"""Fused block-scoring kernel for the M12 decode selector (Triton).

Replaces the eager sequence
    q_g.float(); reps.float(); einsum; amax(-1); amax(G); amax(nq); amax(Hkv)
with one Triton kernel that writes ``scores[B]`` (float32), followed by a
single ``topk``.  The math is identical:

    score[b] = max over nq, h in Hq, j in r of q[nq,h,:] . reps[h//G, b, j, :]

where ``reps`` is ``[Hkv, B, r, D]`` GQA-representative keys and ``G = Hq/Hkv``.

This module is self-contained; it does not modify the engine.  Numerics match
the eager path to float32 accumulation order differences only.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _block_scores_kernel(
    q_ptr, reps_ptr, scores_ptr,
    stride_qn, stride_qh, stride_qd,
    stride_rh, stride_rb, stride_rr, stride_rd,
    nblocks,
    HQ: tl.constexpr, HKV: tl.constexpr, R: tl.constexpr, D: tl.constexpr,
    NQ: tl.constexpr,
):
    b = tl.program_id(0)
    if b >= nblocks:
        return
    G: tl.constexpr = HQ // HKV
    h = tl.arange(0, HQ)
    kv = h // G
    d = tl.arange(0, D)
    j = tl.arange(0, R)

    rep_ptrs = (reps_ptr
                + kv[:, None, None] * stride_rh
                + b * stride_rb
                + j[None, :, None] * stride_rr
                + d[None, None, :] * stride_rd)
    rv = tl.load(rep_ptrs).to(tl.float32)          # [HQ, R, D]

    q_ptrs = q_ptr + h[:, None] * stride_qh + d[None, :] * stride_qd
    acc = tl.full([HQ, R], float("-inf"), tl.float32)
    for i in range(NQ):
        qv = tl.load(q_ptrs + i * stride_qn).to(tl.float32)   # [HQ, D]
        dot = tl.sum(qv[:, None, :] * rv, axis=2)             # [HQ, R]
        acc = tl.maximum(acc, dot)
    tl.store(scores_ptr + b, tl.max(acc))


def fused_block_scores(
    q: torch.Tensor,          # [Hq, D] or [Nq, Hq, D], bf16/fp16/fp32, CUDA
    reps: torch.Tensor,       # [Hkv, B, R, D], contiguous, CUDA
    nblocks: int,
    num_heads: int,
) -> torch.Tensor:
    """Return ``scores [B]`` float32 on the same device (unmasked)."""
    if q.dim() == 2:
        q = q.unsqueeze(0)
    nq, hq, d = q.shape
    hkv, B, R, D = reps.shape
    assert d == D and hq == num_heads and hkv * (hq // hkv) == hq
    q = q.contiguous()
    scores = torch.empty(nblocks, dtype=torch.float32, device=q.device)
    grid = (nblocks,)
    _block_scores_kernel[grid](
        q, reps, scores,
        q.stride(0), q.stride(1), q.stride(2),
        reps.stride(0), reps.stride(1), reps.stride(2), reps.stride(3),
        nblocks,
        HQ=hq, HKV=hkv, R=R, D=D, NQ=nq,
    )
    return scores
