from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from nanovllm.kvdb.prefix_index import PrefixIndex
from nanovllm.kvdb.store.cpu import CPUBlockStore
from nanovllm.kvdb.types import (
    CPUBlockHandle,
    CacheKey,
    CacheTier,
    KVBlockPayload,
    LookupResult,
)


@dataclass(slots=True)
class ContextSession:
    token_ids: tuple[int, ...]
    lookup: LookupResult
    block_size: int

    @property
    def matched_blocks(self) -> int:
        return self.lookup.matched_blocks

    @property
    def matched_tokens(self) -> int:
        return self.lookup.matched_tokens

    @property
    def uncached_token_ids(self) -> tuple[int, ...]:
        return self.token_ids[self.matched_tokens :]

    @property
    def full_block_count(self) -> int:
        return len(self.token_ids) // self.block_size


class ContextDB:
    """Token-prefix coordinator over a CPUBlockStore and PrefixIndex."""

    def __init__(self, store: CPUBlockStore, index: PrefixIndex | None = None) -> None:
        self.store = store
        self.fingerprint = store.fingerprint
        self.block_size = self.fingerprint.block_size
        self.index = index if index is not None else PrefixIndex(self.block_size)
        if self.index.block_size != self.block_size:
            raise ValueError("PrefixIndex block size does not match store fingerprint")
        self.store.add_invalidation_callback(self.index.remove_handle)
        self._requests = 0
        self._hits = 0
        self._misses = 0

    def create_session(
        self, token_ids: Iterable[int], *, record_stats: bool = True
    ) -> ContextSession:
        tokens = tuple(token_ids)
        while True:
            lookup = self.index.lookup(tokens, self.fingerprint)
            stale = [
                handle
                for handle in lookup.cpu_handles
                if not self.store.contains(handle)
            ]
            if not stale:
                break
            for handle in stale:
                self.index.remove_handle(handle)
        for handle in lookup.cpu_handles:
            self.store.touch(handle)
        if record_stats:
            self._requests += 1
            if lookup.matched_blocks:
                self._hits += 1
            else:
                self._misses += 1
        return ContextSession(tokens, lookup, self.block_size)

    def store_session(
        self,
        session: ContextSession,
        uncached_full_block_payloads: Sequence[KVBlockPayload],
    ) -> tuple[CPUBlockHandle, ...]:
        full_block_count = len(session.token_ids) // self.block_size
        expected = full_block_count - session.matched_blocks
        if len(uncached_full_block_payloads) != expected:
            raise ValueError(
                f"expected {expected} uncached full-block payloads, "
                f"got {len(uncached_full_block_payloads)}"
            )
        prefix_hash = -1
        payload_index = 0
        for block_index in range(full_block_count):
            start = block_index * self.block_size
            block_tokens = session.token_ids[start : start + self.block_size]
            prefix_hash = self.index.hash_fn(block_tokens, prefix_hash)
            if block_index < session.matched_blocks:
                continue
            key = CacheKey(self.fingerprint.digest, prefix_hash)
            payload = uncached_full_block_payloads[payload_index]
            payload_index += 1
            handle = self.store.store_block(payload, key=key)
            self.index.register(
                self.fingerprint,
                prefix_hash,
                block_tokens,
                handle,
            )
        # Capacity eviction may have invalidated an earlier block while later
        # blocks were being stored. Re-query so the session only reports the
        # still-contiguous, live prefix rather than stale local handles.
        session.lookup = self.index.lookup(session.token_ids, self.fingerprint)
        if session.lookup.cpu_handles:
            session.lookup = LookupResult(
                matched_blocks=session.lookup.matched_blocks,
                matched_tokens=session.lookup.matched_tokens,
                tier=CacheTier.CPU,
                cpu_handles=session.lookup.cpu_handles,
            )
        return session.lookup.cpu_handles

    def stats(self) -> dict:
        return {
            "requests": self._requests,
            "hits": self._hits,
            "misses": self._misses,
            "index_entries": len(self.index),
            "store": self.store.stats(),
        }
