import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.kvdb.metrics import CacheMetrics
from nanovllm.kvdb.metrics import StageTimer


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config) if field.init}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.metrics = CacheMetrics() if config.enable_cache_metrics else None
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.model_runner.metrics = self.metrics
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config, self.metrics)
        self._ttft_recorded = False
        self._last_cpu_restore_error = None
        atexit.register(self.exit)

    def exit(self):
        # Idempotent: atexit may call this after an explicit shutdown.
        if not hasattr(self, "model_runner"):
            return
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        if self.model_runner.context_db is not None:
            timer = StageTimer()
            timer.__enter__()
            lookup = self.model_runner.lookup_cpu_context(prompt)
            max_reusable_blocks = max((len(prompt) - 1) // Sequence.block_size, 0)
            seq.cpu_cache_handles = lookup.cpu_handles[:max_reusable_blocks]
            seq.cpu_lookup_time_ms = timer.elapsed_ms()
        self.scheduler.add(seq)

    def step(self):
        timer = StageTimer() if self.metrics is not None else None
        if timer is not None:
            timer.__enter__()
        seqs, is_prefill = self.scheduler.schedule()
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        pending_restores = [seq for seq in seqs if seq.cpu_restore_pending]
        if pending_restores:
            try:
                self.model_runner.restore_cpu_blocks(pending_restores)
                for seq in pending_restores:
                    self.scheduler.confirm_cpu_restore(seq)
            except Exception as exc:
                self._last_cpu_restore_error = repr(exc)
                for seq in pending_restores:
                    self.scheduler.rollback_cpu_restore(seq)
                return self.step()
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        if is_prefill:
            self.model_runner.persist_cpu_contexts(seqs)
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        if self.metrics is not None and is_prefill and not self._ttft_recorded:
            self.metrics.record_ttft(timer.elapsed_ms())
            self._ttft_recorded = True
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def get_cache_metrics(self) -> dict:
        """Return a JSON-serializable metrics snapshot for the last run."""
        return {} if self.metrics is None else self.metrics.snapshot()

    def reset_cache_metrics(self) -> None:
        if self.metrics is not None:
            self.metrics.reset()
            self._ttft_recorded = False

    def clear_gpu_prefix_cache(self) -> int:
        if not self.is_finished():
            raise RuntimeError("GPU prefix cache can only be cleared while idle")
        return self.scheduler.block_manager.evict_free_cached_blocks()

    def get_cpu_cache_stats(self) -> dict:
        return self.model_runner.cpu_cache_stats()

    def get_last_cpu_restore_error(self) -> str | None:
        return self._last_cpu_restore_error

    def get_last_cpu_store_error(self) -> str | None:
        return self.model_runner.last_cpu_store_error

    # -- M9 real-model attention trace (opt-in, one-shot) -------------------
    def arm_attention_trace(self, layer_id: int, query_samples: int = 0) -> None:
        """Capture one layer's next cold, single-sequence prefill attention."""
        self.model_runner.call("arm_attention_trace", layer_id, query_samples)

    def retrieve_attention_trace(self):
        """Return the captured CPU AttentionTrace, or None if not captured."""
        return self.model_runner.call("retrieve_attention_trace")

    def clear_attention_trace(self) -> None:
        self.model_runner.call("clear_attention_trace")

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        if self.metrics is not None:
            # Public snapshots describe exactly one generate() call. Keeping a
            # per-call window avoids mixing cumulative lookup counters with a
            # TTFT value that is meaningful only for the current request batch.
            self.reset_cache_metrics()
        self._last_cpu_restore_error = None
        self.model_runner.last_cpu_store_error = None
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
