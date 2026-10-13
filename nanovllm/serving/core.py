"""Bounded FIFO admission around NanoKV's single-sequence sparse engine.

The worker owns model creation, tokenization, CUDA calls and cancellation.
HTTP threads never touch the engine or share its pinned staging buffers.
"""
from __future__ import annotations

import math
import queue
import threading
import time
import uuid
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Callable

from nanovllm.sampling_params import SamplingParams


class ServiceError(Exception):
    def __init__(self, message: str, status: int = 503):
        super().__init__(message)
        self.status = status


@dataclass
class Job:
    messages: list[dict]
    max_tokens: int
    temperature: float
    timeout_s: float
    id: str = field(default_factory=lambda: "chatcmpl-" + uuid.uuid4().hex)
    submitted: float = field(default_factory=time.monotonic)
    cancelled: threading.Event = field(default_factory=threading.Event)
    events: queue.Queue = field(default_factory=queue.Queue)


class EngineAdapter:
    """Real NanoKV adapter; heavy imports happen only in the owner worker."""
    def __init__(self, model: str, options: dict):
        from dataclasses import fields
        from nanovllm.config import Config
        from nanovllm import LLM

        allowed = {f.name for f in fields(Config) if f.init} - {"model"}
        unknown = set(options) - allowed
        if unknown:
            raise ValueError(f"unknown engine options: {sorted(unknown)}")
        self.engine = LLM(model, **options)
        self.max_model_len = self.engine.model_runner.config.max_model_len
        import torch
        self.metadata = {"model_path": model, "options": dict(options),
                         "max_model_len": self.max_model_len, "torch": torch.__version__,
                         "cuda_runtime": torch.version.cuda,
                         "gpu": torch.cuda.get_device_name(0)}

    def encode(self, messages):
        return self.engine.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
        )

    def decode(self, ids):
        return self.engine.tokenizer.decode(
            ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )

    def add(self, ids, params):
        return self.engine.add_request(ids, params)

    def step(self):
        self.engine.step()

    def cancel(self, seq):
        return self.engine.cancel_request(seq.seq_id)

    def close(self):
        self.engine.exit()


class Service:
    def __init__(self, factory: Callable, max_outstanding=4, max_output_tokens=256,
                 max_prompt_tokens=131072, queue_timeout_s=60.0):
        if min(max_outstanding, max_output_tokens, max_prompt_tokens) < 1:
            raise ValueError("capacity and token limits must be positive")
        if not math.isfinite(queue_timeout_s) or queue_timeout_s <= 0:
            raise ValueError("queue timeout must be finite and positive")
        self.factory = factory
        self.max_outstanding = max_outstanding
        self.max_output_tokens = max_output_tokens
        self.max_prompt_tokens = max_prompt_tokens
        self.queue_timeout_s = queue_timeout_s
        self.pending = queue.Queue()
        self.lock = threading.Lock()
        self.ready = threading.Event()
        self.stopping = threading.Event()
        self.healthy = False
        self.startup_error = None
        self.engine_metadata = {}
        self.outstanding = 0
        self.active = None
        self.counts = Counter()
        self.history = deque(maxlen=1024)
        self.thread = threading.Thread(target=self._worker, name="nanokv-engine", daemon=True)

    def start(self, timeout_s=300):
        self.thread.start()
        if not self.ready.wait(timeout_s):
            self.stopping.set()
            raise ServiceError("model initialization timed out")
        if self.startup_error:
            raise ServiceError(f"model initialization failed: {self.startup_error}")

    def submit(self, messages, max_tokens=64, temperature=0.0, timeout_s=300.0):
        if not isinstance(messages, list) or not messages or any(
            not isinstance(m, dict) or m.get("role") not in ("system", "user", "assistant")
            or not isinstance(m.get("content"), str) for m in messages
        ):
            raise ServiceError("messages must contain text system/user/assistant messages", 400)
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or not 1 <= max_tokens <= self.max_output_tokens:
            raise ServiceError(f"max_tokens must be in [1, {self.max_output_tokens}]", 400)
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature < 0:
            raise ServiceError("temperature must be finite and nonnegative", 400)
        if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or not math.isfinite(timeout_s) or not 0 < timeout_s <= 3600:
            raise ServiceError("timeout_s must be in (0, 3600]", 400)
        # Bound payload memory even before expensive model tokenization.
        if sum(len(m["content"]) for m in messages) > self.max_prompt_tokens * 16:
            raise ServiceError("request text exceeds the admission limit", 413)
        with self.lock:
            if not self.healthy or self.stopping.is_set():
                self.counts["unavailable"] += 1
                raise ServiceError("engine unavailable")
            if self.outstanding >= self.max_outstanding:
                self.counts["rejected"] += 1
                raise ServiceError("admission capacity reached", 429)
            job = Job([dict(m) for m in messages], max_tokens, temperature, timeout_s)
            self.outstanding += 1
            self.counts["accepted"] += 1
            self.pending.put(job)
            return job

    def snapshot(self):
        with self.lock:
            return {"healthy": self.healthy, "execution_mode": "single_sequence_fifo",
                    "engine": dict(self.engine_metadata),
                    "outstanding": self.outstanding, "active": int(self.active is not None),
                    "queued": self.outstanding - int(self.active is not None),
                    "capacity": self.max_outstanding, "counts": dict(self.counts),
                    "recent": list(self.history)}

    def metrics(self):
        s = self.snapshot()
        lines = [f'nanokv_requests_total{{status="{k}"}} {v}' for k, v in s["counts"].items()]
        lines += [f"nanokv_{k} {s[k]}" for k in ("outstanding", "active", "queued", "capacity")]
        lines.append(f"nanokv_healthy {int(s['healthy'])}")
        for key in ("queue_ms", "ttft_ms", "tpot_ms", "e2e_ms"):
            xs = sorted(r[key] for r in s["recent"] if r.get(key) is not None and r["status"] == "success")
            if xs:
                for label, fraction in (("0.5", .5), ("0.95", .95), ("0.99", .99)):
                    lines.append(f'nanokv_{key}{{quantile="{label}"}} {xs[math.ceil(fraction * len(xs)) - 1]:.6f}')
        return "\n".join(lines) + "\n"

    def close(self, timeout_s=5):
        self.stopping.set()
        with self.lock:
            self.healthy = False
            if self.active:
                self.active.cancelled.set()
        if self.thread.is_alive():
            self.thread.join(timeout_s)

    def _finish(self, job, status, error=None, **stats):
        with self.lock:
            self.outstanding -= 1
            self.active = None
            self.counts[status] += 1
            self.history.append({"id": job.id, "status": status, **stats})
        job.events.put({"type": "done" if status == "success" else "error",
                        "status": status, "message": error, **stats})

    def _worker(self):
        engine = None
        try:
            engine = self.factory()
            with self.lock:
                self.healthy = not self.stopping.is_set()
                self.engine_metadata = getattr(engine, "metadata", {})
        except Exception as exc:
            self.startup_error = str(exc)
        finally:
            self.ready.set()
        if engine is None:
            return
        try:
            while not self.stopping.is_set():
                try:
                    job = self.pending.get(timeout=.05)
                except queue.Empty:
                    continue
                with self.lock:
                    self.active = job
                started = time.monotonic()
                queue_ms = (started - job.submitted) * 1000
                if job.cancelled.is_set():
                    self._finish(job, "cancelled", "cancelled before execution", queue_ms=queue_ms)
                    continue
                if queue_ms > min(self.queue_timeout_s, job.timeout_s) * 1000:
                    self._finish(job, "timeout", "queue deadline exceeded", queue_ms=queue_ms)
                    continue
                try:
                    ids = engine.encode(job.messages)
                    if not ids or len(ids) > self.max_prompt_tokens or len(ids) + job.max_tokens > engine.max_model_len:
                        raise ValueError("prompt plus max_tokens exceeds the engine context limit")
                except Exception as exc:
                    self._finish(job, "invalid", str(exc), queue_ms=queue_ms)
                    continue
                seq = None
                try:
                    seq = engine.add(ids, SamplingParams(temperature=job.temperature, max_tokens=job.max_tokens))
                    first = last = None
                    emitted = 0
                    sent_text = ""
                    while not seq.is_finished:
                        if job.cancelled.is_set() or self.stopping.is_set():
                            engine.cancel(seq)
                            self._finish(job, "cancelled", "request cancelled", queue_ms=queue_ms)
                            break
                        if time.monotonic() - job.submitted > job.timeout_s:
                            engine.cancel(seq)
                            self._finish(job, "timeout", "request deadline exceeded", queue_ms=queue_ms)
                            break
                        engine.step()
                        completion = seq.completion_token_ids
                        if len(completion) > emitted:
                            now = time.monotonic()
                            first = now if first is None else first
                            last = now
                            decoded = engine.decode(completion)
                            text = decoded if seq.is_finished else decoded.rstrip("\ufffd")
                            if not text.startswith(sent_text):
                                raise RuntimeError("tokenizer streaming output changed an already emitted prefix")
                            job.events.put({"type": "token", "token_ids": completion[emitted:],
                                            "text": text[len(sent_text):], "time": now})
                            emitted, sent_text = len(completion), text
                    else:
                        if first is None:
                            raise RuntimeError("engine finished without output tokens")
                        self._finish(job, "success", prompt_tokens=len(ids), output_tokens=emitted,
                                     finish_reason="length" if emitted == job.max_tokens else "stop",
                                     queue_ms=queue_ms, ttft_ms=(first - job.submitted) * 1000,
                                     tpot_ms=(last - first) * 1000 / (emitted - 1) if emitted > 1 else None,
                                     e2e_ms=(last - job.submitted) * 1000)
                except Exception as exc:
                    # CUDA/runtime failures may leave shared state invalid. Fail
                    # closed and require process restart, rather than reusing it.
                    with self.lock:
                        self.healthy = False
                    self.stopping.set()
                    self._finish(job, "failed", str(exc), queue_ms=queue_ms)
        finally:
            with self.lock:
                self.healthy = False
            while True:
                try:
                    job = self.pending.get_nowait()
                except queue.Empty:
                    break
                self._finish(job, "unavailable", "engine stopped")
            engine.close()
