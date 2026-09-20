from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import xxhash

from nanovllm.kvdb.types import (
    CPUBlockHandle,
    CacheFingerprint,
    CacheKey,
    CacheTier,
    LookupResult,
)


def compute_block_hash(token_ids: Iterable[int], prefix: int = -1) -> int:
    """Match nano-vLLM's upstream chained xxHash semantics exactly."""

    h = xxhash.xxh64()
    if prefix != -1:
        h.update(prefix.to_bytes(8, "little"))
    h.update(np.array(list(token_ids)).tobytes())
    return h.intdigest()


@dataclass(frozen=True, slots=True)
class PrefixEntry:
    token_ids: tuple[int, ...]
    handle: CPUBlockHandle


class PrefixIndex:
    """Collision-safe longest-prefix index over complete token blocks."""

    def __init__(
        self,
        block_size: int,
        *,
        hash_fn: Callable[[Iterable[int], int], int] = compute_block_hash,
    ) -> None:
        if block_size < 1:
            raise ValueError("block_size must be positive")
        self.block_size = block_size
        self.hash_fn = hash_fn
        self._entries: dict[CacheKey, list[PrefixEntry]] = defaultdict(list)
        self._handle_entries: dict[
            CPUBlockHandle, set[tuple[CacheKey, tuple[int, ...]]]
        ] = defaultdict(set)

    def register(
        self,
        fingerprint: CacheFingerprint,
        block_hash: int,
        token_ids: Iterable[int],
        handle: CPUBlockHandle,
    ) -> None:
        tokens = tuple(token_ids)
        if len(tokens) != self.block_size:
            raise ValueError("only complete token blocks may be indexed")
        key = CacheKey(fingerprint.digest, block_hash)
        entries = self._entries[key]
        for entry in entries:
            if entry.token_ids == tokens and entry.handle == handle:
                return
        entries.append(PrefixEntry(tokens, handle))
        self._handle_entries[handle].add((key, tokens))

    def remove_handle(
        self, handle: CPUBlockHandle, key_hint: CacheKey | None = None
    ) -> None:
        references = self._handle_entries.pop(handle, set())
        for key, tokens in references:
            entries = self._entries.get(key, [])
            entries = [
                entry
                for entry in entries
                if not (entry.handle == handle and entry.token_ids == tokens)
            ]
            if entries:
                self._entries[key] = entries
            else:
                self._entries.pop(key, None)

    def lookup(
        self, token_ids: Iterable[int], fingerprint: CacheFingerprint
    ) -> LookupResult:
        tokens = list(token_ids)
        prefix_hash = -1
        handles: list[CPUBlockHandle] = []
        for offset in range(0, len(tokens) - self.block_size + 1, self.block_size):
            block = tuple(tokens[offset : offset + self.block_size])
            prefix_hash = self.hash_fn(block, prefix_hash)
            key = CacheKey(fingerprint.digest, prefix_hash)
            match = next(
                (
                    entry
                    for entry in self._entries.get(key, ())
                    if entry.token_ids == block
                ),
                None,
            )
            if match is None:
                break
            handles.append(match.handle)
        matched_blocks = len(handles)
        return LookupResult(
            matched_blocks=matched_blocks,
            matched_tokens=matched_blocks * self.block_size,
            tier=CacheTier.CPU if handles else CacheTier.MISS,
            cpu_handles=tuple(handles),
        )

    def __len__(self) -> int:
        return sum(len(entries) for entries in self._entries.values())
