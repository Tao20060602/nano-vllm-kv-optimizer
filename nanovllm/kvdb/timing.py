from __future__ import annotations

from time import perf_counter


class ModelStageTimer:
    """Synchronised stage timer with a CPU fallback for tests and probes."""

    def __init__(self, enabled: bool, torch_module):
        self.enabled = enabled
        self.torch = torch_module
        self.start_event = None
        self.end_event = None

    def __enter__(self):
        if self.enabled and self.torch.cuda.is_available():
            self.start_event = self.torch.cuda.Event(enable_timing=True)
            self.end_event = self.torch.cuda.Event(enable_timing=True)
            self.start_event.record()
        else:
            self.start = perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        if self.start_event is not None:
            self.end_event.record()
        else:
            self.end = perf_counter()
        return False

    def elapsed_ms(self) -> float:
        if self.start_event is not None:
            self.end_event.synchronize()
            return float(self.start_event.elapsed_time(self.end_event))
        return (self.end - self.start) * 1000.0
