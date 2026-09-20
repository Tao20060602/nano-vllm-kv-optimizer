from __future__ import annotations

from typing import Any

from nanovllm.kvdb.store.base import BlockStore
from nanovllm.kvdb.types import CacheFingerprint, KVBlockPayload


class GPUBlockStore(BlockStore):
    """Adapter over the KV tensor physically owned by ``ModelRunner``.

    A block view is ephemeral: its integer GPU block ID can be reassigned by
    ``BlockManager``. Durable context identity belongs in a later CPU/index tier.
    """

    def __init__(self, kv_cache: Any, fingerprint: CacheFingerprint | None):
        if fingerprint is None:
            raise ValueError("GPUBlockStore requires a cache fingerprint")
        if kv_cache.ndim != 6 or kv_cache.shape[0] != 2:
            raise ValueError(
                "KV cache must have shape [2, layers, blocks, tokens, heads, head_dim]"
            )
        expected = (
            fingerprint.num_layers,
            fingerprint.block_size,
            fingerprint.num_kv_heads,
            fingerprint.head_dim,
        )
        actual = (
            kv_cache.shape[1],
            kv_cache.shape[3],
            kv_cache.shape[4],
            kv_cache.shape[5],
        )
        if actual != expected:
            raise ValueError(f"KV layout {actual} does not match fingerprint {expected}")
        if str(kv_cache.dtype) != fingerprint.dtype:
            raise ValueError(
                f"KV dtype {kv_cache.dtype} does not match fingerprint {fingerprint.dtype}"
            )
        self.kv_cache = kv_cache
        self.fingerprint = fingerprint

    @property
    def capacity_blocks(self) -> int:
        return self.kv_cache.shape[2]

    @property
    def bytes_per_block(self) -> int:
        return self.kv_cache[:, :, 0].numel() * self.kv_cache.element_size()

    def _validate_block_id(self, block_id: int) -> None:
        if not 0 <= block_id < self.capacity_blocks:
            raise IndexError(f"GPU block ID {block_id} is out of range")

    def block_view(self, block_id: int):
        self._validate_block_id(block_id)
        return self.kv_cache[:, :, block_id]

    def layer_cache(self, layer_id: int):
        if not 0 <= layer_id < self.fingerprint.num_layers:
            raise IndexError(f"layer ID {layer_id} is out of range")
        return self.kv_cache[0, layer_id], self.kv_cache[1, layer_id]

    def read_block(self, block_id: int) -> KVBlockPayload:
        return KVBlockPayload(self.fingerprint, self.block_view(block_id))

    def write_block(self, block_id: int, payload: KVBlockPayload) -> None:
        if payload.fingerprint != self.fingerprint:
            raise ValueError("cache fingerprint mismatch")
        destination = self.block_view(block_id)
        if payload.tensor.shape != destination.shape:
            raise ValueError(
                f"payload shape {payload.tensor.shape} does not match {destination.shape}"
            )
        if payload.tensor.dtype != destination.dtype:
            raise ValueError(
                f"payload dtype {payload.tensor.dtype} does not match {destination.dtype}"
            )
        destination.copy_(payload.tensor)
