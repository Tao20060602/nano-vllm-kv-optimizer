from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


@dataclass(slots=True)
class RequestMetrics:
    seq_id: int
    prompt_tokens: int
    started_at: float
    first_token_at: float | None = None
    finished_at: float | None = None
    completion_tokens: int = 0

    @property
    def ttft_seconds(self) -> float | None:
        if self.first_token_at is None:
            return None
        return self.first_token_at - self.started_at

    @property
    def e2e_seconds(self) -> float | None:
        if self.finished_at is None:
            return None
        return self.finished_at - self.started_at

    @property
    def tpot_seconds(self) -> float | None:
        if self.first_token_at is None or self.finished_at is None or self.completion_tokens <= 1:
            return None
        return (self.finished_at - self.first_token_at) / (self.completion_tokens - 1)

    def to_dict(self) -> dict:
        return {
            "seq_id": self.seq_id,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "ttft_ms": None if self.ttft_seconds is None else self.ttft_seconds * 1000,
            "tpot_ms": None if self.tpot_seconds is None else self.tpot_seconds * 1000,
            "e2e_ms": None if self.e2e_seconds is None else self.e2e_seconds * 1000,
        }


@dataclass(slots=True)
class StepMetrics:
    phase: str
    num_sequences: int
    num_tokens: int
    elapsed_seconds: float

    @property
    def throughput(self) -> float:
        return self.num_tokens / self.elapsed_seconds if self.elapsed_seconds > 0 else 0.0

    def to_dict(self) -> dict:
        data = asdict(self)
        data["throughput_tokens_per_second"] = self.throughput
        return data


class EngineMetrics:
    def __init__(self):
        self.requests: dict[int, RequestMetrics] = {}
        self.steps: list[StepMetrics] = []

    def reset(self):
        self.requests.clear()
        self.steps.clear()

    def register_request(self, seq_id: int, prompt_tokens: int, started_at: float):
        self.requests[seq_id] = RequestMetrics(seq_id, prompt_tokens, started_at)

    def record_request_progress(
        self,
        seq_id: int,
        completion_tokens: int,
        is_finished: bool,
        timestamp: float,
    ):
        metrics = self.requests[seq_id]
        if completion_tokens > 0 and metrics.first_token_at is None:
            metrics.first_token_at = timestamp
        metrics.completion_tokens = completion_tokens
        if is_finished:
            metrics.finished_at = timestamp

    def record_step(self, phase: str, num_sequences: int, num_tokens: int, elapsed_seconds: float):
        self.steps.append(StepMetrics(phase, num_sequences, num_tokens, elapsed_seconds))

    @staticmethod
    def _latency_summary(values: list[float]) -> dict:
        return {
            "mean_ms": None if not values else mean(values) * 1000,
            "p50_ms": None if not values else _percentile(values, 0.50) * 1000,
            "p95_ms": None if not values else _percentile(values, 0.95) * 1000,
        }

    def summary(self) -> dict:
        completed = [request for request in self.requests.values() if request.finished_at is not None]
        ttft = [request.ttft_seconds for request in completed if request.ttft_seconds is not None]
        tpot = [request.tpot_seconds for request in completed if request.tpot_seconds is not None]
        e2e = [request.e2e_seconds for request in completed if request.e2e_seconds is not None]

        phase_stats = {}
        for phase in ("prefill", "decode"):
            steps = [step for step in self.steps if step.phase == phase]
            total_tokens = sum(step.num_tokens for step in steps)
            total_seconds = sum(step.elapsed_seconds for step in steps)
            phase_stats[phase] = {
                "steps": len(steps),
                "tokens": total_tokens,
                "seconds": total_seconds,
                "throughput_tokens_per_second": total_tokens / total_seconds if total_seconds > 0 else 0.0,
            }

        return {
            "requests": {
                "total": len(self.requests),
                "completed": len(completed),
            },
            "latency": {
                "ttft": self._latency_summary(ttft),
                "tpot": self._latency_summary(tpot),
                "e2e": self._latency_summary(e2e),
            },
            "phases": phase_stats,
        }

    def snapshot(self) -> dict:
        return {
            "summary": self.summary(),
            "requests": [request.to_dict() for request in self.requests.values()],
            "steps": [step.to_dict() for step in self.steps],
        }
