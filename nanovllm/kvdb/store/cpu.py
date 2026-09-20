from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any, Callable

import torch

from nanovllm.kvdb.types import (
    CPUBlockHandle,
    CacheFingerprint,
    CacheKey,
    KVBlockPayload,
    StoreStats,
)


class StaleCPUBlockHandle(KeyError):
    pass


@dataclass(slots=True)
class _Slot:
    generation: int = 0
    occupied: bool = False
    tensor: Any = None
    key: CacheKey | None = None


InvalidationCallback = Callable[[CPUBlockHandle, CacheKey | None], None]


class CPUBlockStore:
    """Capacity-bounded synchronous CPU KV block store with deterministic LRU."""

    def __init__(
        self,
        fingerprint: CacheFingerprint,
        *,
        capacity_bytes: int,
        pinned: bool = False,
        on_invalidate: InvalidationCallback | None = None,
    ) -> None:
        self.fingerprint = fingerprint
        self.pinned = pinned
        self.on_invalidate = on_invalidate
        dtype_name = fingerprint.dtype.removeprefix("torch.")
        self.dtype = getattr(torch, dtype_name, None)
        if not isinstance(self.dtype, torch.dtype):
            raise ValueError(f"unsupported torch dtype: {fingerprint.dtype}")
        self.block_shape = (
            2,
            fingerprint.num_layers,
            fingerprint.block_size,
            fingerprint.num_kv_heads,
            fingerprint.head_dim,
        )
        self._bytes_per_block = (
            2
            * fingerprint.num_layers
            * fingerprint.block_size
            * fingerprint.num_kv_heads
            * fingerprint.head_dim
            * self.dtype.itemsize
        )
        self._capacity_blocks = capacity_bytes // self._bytes_per_block
        if self._capacity_blocks < 1:
            raise ValueError(
                f"capacity_bytes={capacity_bytes} cannot hold one "
                f"{self._bytes_per_block}-byte KV block"
            )
        self.capacity_bytes = self._capacity_blocks * self._bytes_per_block
        self._slots = [_Slot() for _ in range(self._capacity_blocks)]
        self._free_slot_ids = deque(range(self._capacity_blocks))
        self._lru: OrderedDict[int, None] = OrderedDict()
        self._stats = StoreStats()

    @property
    def capacity_blocks(self) -> int:
        return self._capacity_blocks

    @property
    def bytes_per_block(self) -> int:
        return self._bytes_per_block

    @property
    def resident_blocks(self) -> int:
        return len(self._lru)

    @property
    def resident_bytes(self) -> int:
        return self.resident_blocks * self.bytes_per_block

    def _validate_payload(self, payload: KVBlockPayload) -> None:
        if payload.fingerprint != self.fingerprint:
            raise ValueError("cache fingerprint mismatch")
        if tuple(payload.tensor.shape) != self.block_shape:
            raise ValueError(
                f"payload shape {tuple(payload.tensor.shape)} does not match "
                f"{self.block_shape}"
            )
        if payload.tensor.dtype != self.dtype:
            raise ValueError(
                f"payload dtype {payload.tensor.dtype} does not match {self.dtype}"
            )

    def _resolve(self, handle: CPUBlockHandle, *, touch: bool) -> _Slot:
        if not 0 <= handle.slot_id < self.capacity_blocks:
            raise StaleCPUBlockHandle(f"CPU slot {handle.slot_id} is out of range")
        slot = self._slots[handle.slot_id]
        if not slot.occupied or slot.generation != handle.generation:
            raise StaleCPUBlockHandle(
                f"stale CPU handle ({handle.slot_id}, {handle.generation})"
            )
        if touch:
            self._lru.move_to_end(handle.slot_id)
        return slot

    def _invalidate(self, slot_id: int, *, eviction: bool) -> None:
        slot = self._slots[slot_id]
        handle = CPUBlockHandle(slot_id, slot.generation)
        if self.on_invalidate is not None:
            self.on_invalidate(handle, slot.key)
        slot.occupied = False
        slot.key = None
        if eviction:
            self._stats.evictions += 1

    @staticmethod
    def _synchronize_if_cuda(tensor) -> None:
        if tensor.device.type == "cuda":
            torch.cuda.synchronize(tensor.device)

    def store_block(
        self,
        payload: KVBlockPayload,
        *,
        key: CacheKey | None = None,
    ) -> CPUBlockHandle:
        self._validate_payload(payload)
        source = payload.tensor.detach()

        if self._free_slot_ids:
            slot_id = self._free_slot_ids.popleft()
        else:
            slot_id, _ = self._lru.popitem(last=False)
            self._invalidate(slot_id, eviction=True)
        slot = self._slots[slot_id]
        slot.generation += 1
        if slot.tensor is None:
            slot.tensor = torch.empty(
                self.block_shape,
                dtype=self.dtype,
                device="cpu",
                pin_memory=self.pinned,
            )

        self._synchronize_if_cuda(source)
        start = perf_counter()
        try:
            slot.tensor.copy_(source, non_blocking=False)
            self._synchronize_if_cuda(source)
        except Exception:
            self._free_slot_ids.append(slot_id)
            raise
        elapsed_ms = (perf_counter() - start) * 1000.0
        slot.occupied = True
        slot.key = key
        self._lru[slot_id] = None
        self._stats.stores += 1
        self._stats.store_time_ms += elapsed_ms
        if source.device.type == "cuda":
            self._stats.d2h_bytes += self.bytes_per_block
        return CPUBlockHandle(slot_id, slot.generation)

    def read_block(self, handle: CPUBlockHandle) -> KVBlockPayload:
        slot = self._resolve(handle, touch=True)
        return KVBlockPayload(self.fingerprint, slot.tensor)

    def load_into(self, handle: CPUBlockHandle, destination) -> None:
        slot = self._resolve(handle, touch=True)
        if tuple(destination.shape) != self.block_shape:
            raise ValueError(
                f"destination shape {tuple(destination.shape)} does not match "
                f"{self.block_shape}"
            )
        if destination.dtype != self.dtype:
            raise ValueError(
                f"destination dtype {destination.dtype} does not match {self.dtype}"
            )
        self._synchronize_if_cuda(destination)
        start = perf_counter()
        destination.copy_(slot.tensor, non_blocking=False)
        self._synchronize_if_cuda(destination)
        elapsed_ms = (perf_counter() - start) * 1000.0
        self._stats.loads += 1
        self._stats.load_time_ms += elapsed_ms
        if destination.device.type == "cuda":
            self._stats.h2d_bytes += self.bytes_per_block

    def delete(self, handle: CPUBlockHandle) -> None:
        self._resolve(handle, touch=False)
        self._lru.pop(handle.slot_id)
        self._invalidate(handle.slot_id, eviction=False)
        self._free_slot_ids.append(handle.slot_id)
        self._stats.deletes += 1

    def stats(self) -> dict[str, int | float | bool]:
        result = asdict(self._stats)
        result.update(
            {
                "pinned": self.pinned,
                "capacity_blocks": self.capacity_blocks,
                "capacity_bytes": self.capacity_bytes,
                "resident_blocks": self.resident_blocks,
                "resident_bytes": self.resident_bytes,
                "d2h_bandwidth_gbps": (
                    self._stats.d2h_bytes / self._stats.store_time_ms / 1_000_000
                    if self._stats.store_time_ms
                    else 0.0
                ),
                "h2d_bandwidth_gbps": (
                    self._stats.h2d_bytes / self._stats.load_time_ms / 1_000_000
                    if self._stats.load_time_ms
                    else 0.0
                ),
            }
        )
        return result
