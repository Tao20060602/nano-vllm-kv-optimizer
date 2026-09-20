from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from hashlib import sha256
import json
from typing import Any


@dataclass(frozen=True, slots=True)
class CacheFingerprint:
    """Identity of model semantics and the physical KV layout."""

    model_id: str
    config_digest: str
    dtype: str
    num_layers: int
    num_kv_heads: int
    head_dim: int
    block_size: int
    tensor_parallel_size: int
    rope_theta: float
    rope_scaling: str

    @property
    def digest(self) -> str:
        encoded = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class CacheKey:
    fingerprint_digest: str
    block_hash: int


@dataclass(frozen=True, slots=True)
class GPUBlockHandle:
    """Ephemeral physical GPU identity; never a durable context identity."""

    block_id: int


@dataclass(frozen=True, slots=True)
class CPUBlockHandle:
    """Stable CPU slot identity protected against stale reuse by generation."""

    slot_id: int
    generation: int


@dataclass(slots=True)
class KVBlockPayload:
    fingerprint: CacheFingerprint
    tensor: Any


class CacheTier(str, Enum):
    MISS = "miss"
    GPU = "gpu"
    CPU = "cpu"


@dataclass(frozen=True, slots=True)
class LookupResult:
    matched_blocks: int
    matched_tokens: int
    tier: CacheTier
    gpu_handles: tuple[GPUBlockHandle, ...] = ()
    cpu_handles: tuple[CPUBlockHandle, ...] = ()


@dataclass(slots=True)
class StoreStats:
    stores: int = 0
    loads: int = 0
    deletes: int = 0
    d2h_bytes: int = 0
    h2d_bytes: int = 0
    store_time_ms: float = 0.0
    load_time_ms: float = 0.0
    evictions: int = 0
