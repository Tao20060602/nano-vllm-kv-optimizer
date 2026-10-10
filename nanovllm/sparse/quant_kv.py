"""Quantized CPU KV history for the M12 sparse path (design prototype).

Goal: halve the H2D bytes of the selected remote-history K/V by storing the
per-layer CPU history as int8 instead of bf16, then dequantizing on the GPU
before attention. Sink/recent windows are separate GPU bf16 buffers and are
never quantized.

Scheme (matches the agreed design):
  * K: symmetric int8, **per-channel** scale ``[Hkv, D]``, estimated once from
    an initial window and kept fixed (K channel outliers are stable).
  * V: symmetric int8, **per-block** scale ``[nblocks, Hkv]``, computed per
    64-token block (V outliers move with position).
  * Dequantize on GPU: ``x = q.float() * scale`` (low-precision store,
    high-precision accumulate). Independent elementwise step.

This module is standalone (no engine edits). See bench_logs/lineA for the
numerical test.
"""

from __future__ import annotations

import torch

QMAX = 127


def quantize_k_per_channel(
    k: torch.Tensor,           # [T, Hkv, D] fp32/bf16, the calibration window
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (int8 k, scale [Hkv, D] fp32). Channels = (kv_head, head_dim)."""
    kf = k.float()
    scale = kf.abs().amax(dim=0).clamp_min(1e-8) / QMAX          # [Hkv, D]
    q = torch.round(kf / scale).clamp(-QMAX, QMAX).to(torch.int8)
    return q, scale


def apply_k_scale(k_int8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize K: [..., Hkv, D] int8 * scale[Hkv, D] -> float32."""
    return k_int8.float() * scale


def quantize_v_per_block(
    v: torch.Tensor,           # [nblocks, B, Hkv, D] fp32/bf16
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (int8 v [nblocks,B,Hkv,D], scale [nblocks, Hkv] fp32)."""
    vf = v.float()
    scale = vf.abs().amax(dim=(1, 3)).clamp_min(1e-8) / QMAX      # [nblocks, Hkv]
    q = torch.round(vf / scale[:, None, :, None]).clamp(-QMAX, QMAX).to(torch.int8)
    return q, scale


def dequant_v_rows(
    v_int8: torch.Tensor,      # [S, Hkv, D] gathered rows
    row_scale: torch.Tensor,   # [S, Hkv] scale of the block each row belongs to
) -> torch.Tensor:
    return v_int8.float() * row_scale[:, :, None]


def rows_to_block_scale(scale: torch.Tensor, block_ids: torch.Tensor,
                        block_size: int) -> torch.Tensor:
    """Expand per-block scale [nblocks,Hkv] to per gathered row [S,Hkv]."""
    per_row = scale[block_ids]                                  # [K, Hkv]
    return per_row.repeat_interleave(block_size, dim=0)          # [S, Hkv]
