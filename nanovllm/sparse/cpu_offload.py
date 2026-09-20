"""Synchronous single-layer CPU KV offload laboratory (milestone M9, Route A).

One attention layer's historical K/V live in CPU memory in the original model
dtype/layout ``[num_tokens, num_kv_heads, head_dim]``.  For one decode query we:

1. score all keys on CPU (float32; exact scan = oracle);
2. select fixed top-k or exact Block-DIPR blocks, union across query heads;
3. gather selected rows into *contiguous* CPU staging tensors (pinned on request);
4. copy only the packed selected K/V to GPU (synchronous H2D);
5. run packed sparse attention on GPU.

The full historical K/V is **never** transferred to GPU in the sparse path.
V1 deliberately omits CUDA-stream overlap, double buffering and NUMA tuning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable

import torch

from nanovllm.sparse.block_sparse import (
    exact_block_scores,
    gqa_token_scores,
    selected_token_indices,
    sparse_decode_attention,
    union_block_mask,
)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


@dataclass
class PackedCPUStaging:
    """Contiguous packed K/V for ``S`` selected tokens, held on CPU."""

    indices: torch.Tensor    # [S] int64 CPU, sorted unique
    packed_k: torch.Tensor   # [S, Hkv, D] CPU contiguous, pinned if pinned
    packed_v: torch.Tensor   # [S, Hkv, D] CPU contiguous, pinned if pinned
    pinned: bool

    @property
    def num_selected_tokens(self) -> int:
        return int(self.indices.numel())

    @property
    def packed_k_bytes(self) -> int:
        return self.packed_k.numel() * self.packed_k.element_size()

    @property
    def packed_v_bytes(self) -> int:
        return self.packed_v.numel() * self.packed_v.element_size()

    @property
    def h2d_bytes(self) -> int:
        return self.packed_k_bytes + self.packed_v_bytes


class CPULayerKVStore:
    """Owns one layer's historical K/V in CPU memory (pageable or pinned)."""

    def __init__(self, k: torch.Tensor, v: torch.Tensor, pinned: bool = False):
        if k.dim() != 3 or v.dim() != 3:
            raise ValueError("k/v must be [num_tokens, num_kv_heads, head_dim]")
        if k.shape != v.shape:
            raise ValueError(f"k/v shape mismatch: {k.shape} vs {v.shape}")
        if k.dtype != v.dtype:
            raise ValueError(f"k/v dtype mismatch: {k.dtype} vs {v.dtype}")
        if k.device.type != "cpu" or v.device.type != "cpu":
            raise ValueError("CPULayerKVStore only owns CPU tensors")
        self.num_tokens, self.num_kv_heads, self.head_dim = k.shape
        self.dtype = k.dtype
        # Owned, contiguous copies.
        self.k_cpu = torch.empty_like(k).copy_(k)
        self.v_cpu = torch.empty_like(v).copy_(v)
        self.pinned = bool(pinned)
        if self.pinned:
            if not torch.cuda.is_available():
                raise RuntimeError("pinned memory requires CUDA availability")
            self.k_cpu = self.k_cpu.pin_memory()
            self.v_cpu = self.v_cpu.pin_memory()
        if self.k_cpu.is_pinned() != self.pinned:
            raise RuntimeError("failed to allocate pinned CPU K/V staging")
        if not (self.k_cpu.is_contiguous() and self.v_cpu.is_contiguous()):
            raise RuntimeError("CPU K/V must be contiguous")

    @property
    def resident_bytes(self) -> int:
        """Active bytes of the full K/V held on CPU for this layer."""
        per_tensor = self.num_tokens * self.num_kv_heads * self.head_dim * self.dtype.itemsize
        return 2 * per_tensor

    def full_kv_bytes(self) -> int:
        return self.resident_bytes

    def _validate_indices(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.device.type != "cpu":
            raise ValueError("indices must be CPU tensors")
        if indices.dtype != torch.long:
            indices = indices.long()
        if indices.dim() != 1:
            raise ValueError("indices must be 1-D")
        if indices.numel() == 0:
            raise ValueError("selected index set is empty")
        if int(indices.min()) < 0 or int(indices.max()) >= self.num_tokens:
            raise IndexError("selected indices out of bounds")
        if torch.any(indices[1:] < indices[:-1]):
            raise ValueError("selected indices must be sorted")
        if indices.numel() != torch.unique(indices).numel():
            raise ValueError("selected indices must be unique")
        return indices

    def gather(self, indices: torch.Tensor) -> PackedCPUStaging:
        """Copy selected rows into fresh contiguous (pinned) staging tensors.

        Advanced indexing of a pinned tensor returns pageable memory, so we
        explicitly allocate pinned output and ``index_select`` *into* it.
        """
        indices = self._validate_indices(indices)
        s = indices.numel()
        shape = (s, self.num_kv_heads, self.head_dim)
        packed_k = torch.empty(shape, dtype=self.dtype, device="cpu", pin_memory=self.pinned)
        packed_v = torch.empty(shape, dtype=self.dtype, device="cpu", pin_memory=self.pinned)
        torch.index_select(self.k_cpu, 0, indices, out=packed_k)
        torch.index_select(self.v_cpu, 0, indices, out=packed_v)
        if not (packed_k.is_contiguous() and packed_v.is_contiguous()):
            raise RuntimeError("packed staging must be contiguous")
        if packed_k.is_pinned() != self.pinned or packed_v.is_pinned() != self.pinned:
            raise RuntimeError("packed staging pinning does not match store mode")
        return PackedCPUStaging(indices, packed_k, packed_v, self.pinned)


@dataclass
class RouteAResult:
    output: torch.Tensor                 # [Hq, D] GPU, model dtype
    indices: torch.Tensor                # [S] CPU int64 sorted unique
    token_scores: torch.Tensor           # [Hq, T] CPU float32 raw scores
    num_selected_blocks: int
    num_selected_tokens: int
    num_tokens: int
    timings_ms: dict = field(default_factory=dict)
    packed_k_bytes: int = 0
    packed_v_bytes: int = 0
    h2d_bytes: int = 0
    full_kv_bytes: int = 0
    active_byte_ratio: float = 0.0
    pinned: bool = False


def route_a_replay(
    store: CPULayerKVStore,
    q_cpu: torch.Tensor,
    build_per_head_mask: Callable[[torch.Tensor], torch.Tensor],
    device: torch.device | str,
    retrieval_block_size: int = 64,
    first_tokens: int = 0,
    recent_tokens: int = 128,
    scale: float | None = None,
) -> RouteAResult:
    """Run one synchronous Route-A offload replay and return result + timings."""
    device = torch.device(device)
    if q_cpu.device.type != "cpu":
        q_cpu = q_cpu.cpu()
    if scale is None:
        scale = store.head_dim ** -0.5
    t = store.num_tokens

    total_start = perf_counter()

    # --- 1-4. CPU exact scoring + selection (float32) ---------------------
    search_start = perf_counter()
    q_f32 = q_cpu.float()
    k_f32 = store.k_cpu.float()
    token_scores = gqa_token_scores(q_f32, k_f32)          # [Hq, T]
    block_scores = exact_block_scores(token_scores, retrieval_block_size)
    per_head_mask = build_per_head_mask(block_scores)
    union = union_block_mask(per_head_mask)
    indices = selected_token_indices(
        union, t, retrieval_block_size,
        first_tokens=first_tokens, recent_tokens=recent_tokens,
    )
    search_ms = (perf_counter() - search_start) * 1000.0

    # --- 5. CPU gather into contiguous staging ----------------------------
    gather_start = perf_counter()
    staging = store.gather(indices)
    gather_ms = (perf_counter() - gather_start) * 1000.0

    # Tiny query H2D (not part of the reported K/V H2D bytes/timing).
    q_gpu = q_cpu.to(device, non_blocking=False)

    # --- 6. synchronous H2D of packed selected K/V only -------------------
    _sync(device)
    h2d_start = perf_counter()
    packed_k_gpu = staging.packed_k.to(device, non_blocking=False)
    packed_v_gpu = staging.packed_v.to(device, non_blocking=False)
    _sync(device)
    h2d_ms = (perf_counter() - h2d_start) * 1000.0

    # --- 7. GPU packed sparse attention -----------------------------------
    _sync(device)
    attn_start = perf_counter()
    packed_positions = torch.arange(staging.num_selected_tokens, device=device)
    output = sparse_decode_attention(
        q_gpu, packed_k_gpu, packed_v_gpu, packed_positions, scale=scale
    )
    _sync(device)
    attn_ms = (perf_counter() - attn_start) * 1000.0

    total_ms = (perf_counter() - total_start) * 1000.0

    h2d_bytes = staging.h2d_bytes
    full_bytes = store.full_kv_bytes()
    expected_h2d = (
        2 * staging.num_selected_tokens * store.num_kv_heads
        * store.head_dim * store.dtype.itemsize
    )
    if h2d_bytes != expected_h2d:
        raise RuntimeError(
            f"H2D byte accounting mismatch: {h2d_bytes} != {expected_h2d}"
        )
    active_ratio = staging.num_selected_tokens / t
    if abs(h2d_bytes / full_bytes - active_ratio) > 1e-9:
        raise RuntimeError("packed/full byte ratio does not equal S/num_tokens")

    return RouteAResult(
        output=output,
        indices=staging.indices,
        token_scores=token_scores,
        num_selected_blocks=int(union.sum().item()),
        num_selected_tokens=staging.num_selected_tokens,
        num_tokens=t,
        timings_ms={
            "cpu_search_ms": search_ms,
            "cpu_gather_ms": gather_ms,
            "h2d_ms": h2d_ms,
            "gpu_packed_attention_ms": attn_ms,
            "total_replay_ms": total_ms,
        },
        packed_k_bytes=staging.packed_k_bytes,
        packed_v_bytes=staging.packed_v_bytes,
        h2d_bytes=h2d_bytes,
        full_kv_bytes=full_bytes,
        active_byte_ratio=active_ratio,
        pinned=store.pinned,
    )


from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable

import torch

from nanovllm.sparse.block_sparse import (
    selected_token_indices,
    sparse_decode_attention,
)


@dataclass
class SelectorRouteResult:
    output: torch.Tensor
    indices: torch.Tensor
    union_blocks: torch.Tensor
    num_selected_blocks: int
    num_selected_tokens: int
    num_tokens: int
    timings_ms: dict
    work: dict
    h2d_bytes: int
    full_kv_bytes: int
    active_byte_ratio: float
    pinned: bool


def route_a_selective(
    store,
    q_cpu: torch.Tensor,
    run_selector: Callable[[], dict],
    device,
    retrieval_block_size: int = 64,
    first_tokens: int = 0,
    recent_tokens: int = 128,
    scale=None,
) -> SelectorRouteResult:
    """One timed approximate Route-A replay driven by an arbitrary selector.

    ``run_selector`` is a zero-arg callable returning a dict with keys
    ``union_mask`` (bool [B] CPU, before forced windows), ``search_ms`` and
    ``refine_ms`` (CPU) plus arbitrary ``work`` counters.  The whole replay is
    timed around one complete run; the approximate path never scans all keys
    and only the packed selected K/V crosses to CUDA.
    """
    device = torch.device(device)
    if q_cpu.device.type != "cpu":
        q_cpu = q_cpu.cpu()
    if scale is None:
        scale = store.head_dim ** -0.5
    t = store.num_tokens

    total_start = perf_counter()

    sel = run_selector()
    union_mask = sel["union_mask"]
    search_ms = float(sel["search_ms"])
    refine_ms = float(sel["refine_ms"])
    work = dict(sel.get("work", {}))

    indices = selected_token_indices(
        union_mask, t, retrieval_block_size,
        first_tokens=first_tokens, recent_tokens=recent_tokens,
    )

    gather_start = perf_counter()
    staging = store.gather(indices)
    if device.type == "cuda":
        torch.cuda.synchronize()
    gather_ms = (perf_counter() - gather_start) * 1000.0

    q_gpu = q_cpu.to(device, non_blocking=False)

    if device.type == "cuda":
        torch.cuda.synchronize()
    h2d_start = perf_counter()
    packed_k_gpu = staging.packed_k.to(device, non_blocking=False)
    packed_v_gpu = staging.packed_v.to(device, non_blocking=False)
    if device.type == "cuda":
        torch.cuda.synchronize()
    h2d_ms = (perf_counter() - h2d_start) * 1000.0

    if device.type == "cuda":
        torch.cuda.synchronize()
    attn_start = perf_counter()
    packed_positions = torch.arange(staging.num_selected_tokens, device=device)
    output = sparse_decode_attention(
        q_gpu, packed_k_gpu, packed_v_gpu, packed_positions, scale=scale
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    attn_ms = (perf_counter() - attn_start) * 1000.0

    total_ms = (perf_counter() - total_start) * 1000.0

    h2d_bytes = staging.h2d_bytes
    full_bytes = store.full_kv_bytes()
    expected_h2d = (
        2 * staging.num_selected_tokens * store.num_kv_heads
        * store.head_dim * store.dtype.itemsize
    )
    if h2d_bytes != expected_h2d:
        raise RuntimeError(f"H2D byte accounting mismatch: {h2d_bytes} != {expected_h2d}")
    active_ratio = staging.num_selected_tokens / t

    return SelectorRouteResult(
        output=output,
        indices=staging.indices,
        union_blocks=union_mask,
        num_selected_blocks=int(union_mask.sum().item()),
        num_selected_tokens=staging.num_selected_tokens,
        num_tokens=t,
        timings_ms={
            "approx_search_ms": search_ms,
            "approx_refine_ms": refine_ms,
            "cpu_gather_ms": gather_ms,
            "h2d_ms": h2d_ms,
            "gpu_packed_attention_ms": attn_ms,
            "total_replay_ms": total_ms,
        },
        work=work,
        h2d_bytes=h2d_bytes,
        full_kv_bytes=full_bytes,
        active_byte_ratio=active_ratio,
        pinned=store.pinned,
    )
