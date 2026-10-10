"""Fused dequantize-and-pack kernel for quantized history (int8 -> model dtype).

Replaces the eager per-layer sequence

    hist_k_q.float() * k_scale               -> packed_k (bf16)
    hist_v_q.float() * row_scale[...]        -> packed_v (bf16)

(which allocates fp32 temporaries, does a ``repeat_interleave`` for the V
per-block scale, and launches several kernels) with ONE Triton kernel that
reads int8 plus scales and writes the packed model-dtype buffers directly.

K scale is per-(kv_head, head_dim); V scale is per-(block, kv_head); the
block of a packed row is ``row // block_size``.  Each program handles one
selected history row (Hkv*D elements) and writes both K and V.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _dequant_pack_kernel(
    kq_ptr, vq_ptr, ks_ptr, vs_ptr, ok_ptr, ov_ptr,
    S, block_size,
    HKV: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= S:
        return
    offs = tl.arange(0, BLOCK)
    mask = offs < HKV * D
    base = row * (HKV * D)

    kq = tl.load(kq_ptr + base + offs, mask=mask, other=0).to(tl.float32)
    ks = tl.load(ks_ptr + offs, mask=mask, other=0.0)          # [HKV*D] flat
    tl.store(ok_ptr + base + offs, (kq * ks).to(ok_ptr.dtype.element_ty), mask=mask)

    vq = tl.load(vq_ptr + base + offs, mask=mask, other=0).to(tl.float32)
    blk = row // block_size
    vs = tl.load(vs_ptr + blk * HKV + (offs // D), mask=mask, other=0.0)
    tl.store(ov_ptr + base + offs, (vq * vs).to(ov_ptr.dtype.element_ty), mask=mask)


def fused_dequant_pack(
    hist_k_q: torch.Tensor,     # [S, Hkv, D] int8, CUDA
    hist_v_q: torch.Tensor,     # [S, Hkv, D] int8, CUDA
    k_scale: torch.Tensor,      # [Hkv, D] fp32, CUDA
    v_scale: torch.Tensor,      # [K, Hkv] fp32, CUDA (per selected block)
    packed_k: torch.Tensor,     # [S, Hkv, D] model dtype, CUDA (write)
    packed_v: torch.Tensor,     # [S, Hkv, D] model dtype, CUDA (write)
    block_size: int,
) -> None:
    S, hkv, d = hist_k_q.shape
    BLOCK = triton.next_power_of_2(hkv * d)
    grid = (S,)
    _dequant_pack_kernel[grid](
        hist_k_q, hist_v_q, k_scale.reshape(-1), v_scale, packed_k, packed_v,
        S, block_size,
        HKV=hkv, D=d, BLOCK=BLOCK,
    )
