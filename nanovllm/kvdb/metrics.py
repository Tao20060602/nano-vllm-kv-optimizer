from __future__ import annotations

from dataclasses import dataclass, field, fields
from threading import Lock
from time import perf_counter
from typing import Any


@dataclass(slots=True)
class CacheMetrics:
    """Structured counters and timings for one engine instance.

    Values are intentionally plain Python scalars so callers can serialize the
    snapshot directly to JSON. Timings are milliseconds and byte counters are
    physical transfer bytes, not estimates.
    """

    requests: int = 0
    hits: int = 0
    misses: int = 0
    gpu_hit_blocks: int = 0
    cpu_hit_blocks: int = 0
    reused_tokens: int = 0
    recomputed_tokens: int = 0
    prefill_executed_tokens: int = 0
    decode_executed_tokens: int = 0
    lookup_time_ms: float = 0.0
    load_time_ms: float = 0.0
    store_time_ms: float = 0.0
    prefill_time_ms: float = 0.0
    first_token_time_ms: float = 0.0
    decode_time_ms: float = 0.0
    ttft_ms: float = 0.0
    h2d_bytes: int = 0
    d2h_bytes: int = 0
    eviction_count: int = 0
    cpu_load_failures: int = 0
    cpu_store_failures: int = 0
    _lock: Lock = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._lock = Lock()

    def record_lookup(
        self,
        matched_blocks: int,
        matched_tokens: int,
        total_tokens: int,
        tier: str = "gpu",
    ) -> None:
        with self._lock:
            self.requests += 1
            self.reused_tokens += matched_tokens
            self.recomputed_tokens += max(total_tokens - matched_tokens, 0)
            if matched_blocks:
                self.hits += 1
                if tier == "cpu":
                    self.cpu_hit_blocks += matched_blocks
                else:
                    self.gpu_hit_blocks += matched_blocks
            else:
                self.misses += 1

    def record_lookup_time(self, elapsed_ms: float) -> None:
        with self._lock:
            self.lookup_time_ms += elapsed_ms

    def record_prefill(self, elapsed_ms: float) -> None:
        with self._lock:
            self.prefill_time_ms += elapsed_ms

    def record_executed_tokens(self, token_count: int, *, is_prefill: bool) -> None:
        with self._lock:
            if is_prefill:
                self.prefill_executed_tokens += token_count
            else:
                self.decode_executed_tokens += token_count

    def record_ttft(self, elapsed_ms: float) -> None:
        with self._lock:
            self.first_token_time_ms += elapsed_ms
            self.ttft_ms += elapsed_ms

    def record_decode(self, elapsed_ms: float) -> None:
        with self._lock:
            self.decode_time_ms += elapsed_ms

    def record_load(self, elapsed_ms: float, h2d_bytes: int) -> None:
        with self._lock:
            self.load_time_ms += elapsed_ms
            self.h2d_bytes += h2d_bytes

    def record_store(
        self, elapsed_ms: float, d2h_bytes: int, evictions: int = 0
    ) -> None:
        with self._lock:
            self.store_time_ms += elapsed_ms
            self.d2h_bytes += d2h_bytes
            self.eviction_count += evictions

    def record_cpu_restore_fallback(
        self, matched_blocks: int, matched_tokens: int
    ) -> None:
        with self._lock:
            self.cpu_load_failures += 1
            self.hits -= 1
            self.misses += 1
            self.cpu_hit_blocks -= matched_blocks
            self.reused_tokens -= matched_tokens
            self.recomputed_tokens += matched_tokens

    def record_cpu_store_failure(self) -> None:
        with self._lock:
            self.cpu_store_failures += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                item.name: getattr(self, item.name)
                for item in fields(self)
                if item.name != "_lock"
            }

    def reset(self) -> None:
        with self._lock:
            for name in self.__dataclass_fields__:
                if name != "_lock":
                    setattr(self, name, 0)


class StageTimer:
    """Small wall-clock fallback used around CPU-side orchestration.

    CUDA execution timing is handled by ``ModelStageTimer`` in model_runner;
    this class exists for lookup timing, which happens before model execution.
    """

    def __enter__(self) -> "StageTimer":
        self.start = perf_counter()
        return self

    def elapsed_ms(self) -> float:
        return (perf_counter() - self.start) * 1000.0
