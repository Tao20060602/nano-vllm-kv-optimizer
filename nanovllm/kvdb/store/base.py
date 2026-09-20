from __future__ import annotations

from abc import ABC, abstractmethod

from nanovllm.kvdb.types import CacheFingerprint, KVBlockPayload


class BlockStore(ABC):
    """Physical KV storage contract; logical allocation stays in BlockManager."""

    fingerprint: CacheFingerprint

    @property
    @abstractmethod
    def capacity_blocks(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def bytes_per_block(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def read_block(self, block_id: int) -> KVBlockPayload:
        raise NotImplementedError

    @abstractmethod
    def write_block(self, block_id: int, payload: KVBlockPayload) -> None:
        raise NotImplementedError
