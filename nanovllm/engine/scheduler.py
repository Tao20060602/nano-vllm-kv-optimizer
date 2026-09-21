from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager
from nanovllm.kvdb.metrics import StageTimer


class Scheduler:

    def __init__(self, config: Config, metrics=None):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.sparse = getattr(config, "enable_sparse_attention", False)
        self.sparse_prefill_chunk_size = getattr(config, 'sparse_prefill_chunk_size', 0)
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size, metrics)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    # -- M11 sparse virtual-allocation path (no physical paged blocks) ---
    def _sparse_schedule(self) -> tuple[list[Sequence], bool]:
        # single sequence only; reject batching/preemption.
        if self.waiting:
            if self.running:
                raise RuntimeError(
                    "sparse mode supports one sequence at a time; "
                    "rejecting a second prompt while one is running")
            if len(self.waiting) > 1:
                raise RuntimeError(
                    "sparse mode rejects batched prompts (waiting>1)")
            seq = self.waiting.popleft()
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)
        elif self.running and len(self.running) == 1:
            seq = self.running[0]
        else:
            raise RuntimeError(
                "sparse mode must have exactly one running sequence")

        # prefill is done when all PROMPT tokens are cached (num_tokens grows
        # during decode via append_token, so it cannot be the completion test)
        if seq.num_cached_tokens < seq.num_prompt_tokens:
            # prefill, possibly chunked (M12 long-context path)
            remaining = seq.num_prompt_tokens - seq.num_cached_tokens
            chunk = getattr(self, "sparse_prefill_chunk_size", 0) or remaining
            seq.num_scheduled_tokens = min(remaining, chunk)
            seq.is_prefill = True
            return [seq], True
        seq.num_scheduled_tokens = 1
        seq.is_prefill = False
        return [seq], False

    def schedule(self) -> tuple[list[Sequence], bool]:
        if self.sparse:
            return self._sparse_schedule()

        scheduled_seqs = []
        num_batched_tokens = 0

        # prefill
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if not seq.block_table:
                timer = StageTimer() if self.block_manager.metrics is not None else None
                if timer is not None:
                    timer.__enter__()
                gpu_cached_blocks = self.block_manager.can_allocate(
                    seq, record_metrics=False
                )
                cpu_cached_blocks = (
                    0 if seq.cpu_restore_failed else len(seq.cpu_cache_handles)
                )
                use_cpu = (
                    cpu_cached_blocks > max(gpu_cached_blocks, 0)
                    and self.block_manager.can_allocate_fresh(seq)
                )
                if use_cpu:
                    num_cached_blocks = cpu_cached_blocks
                else:
                    num_cached_blocks = gpu_cached_blocks
                if num_cached_blocks == -1:
                    break
                if self.block_manager.metrics is not None and not seq.metrics_lookup_recorded:
                    tier = "cpu" if use_cpu else "gpu"
                    self.block_manager.metrics.record_lookup(
                        num_cached_blocks,
                        num_cached_blocks * self.block_size,
                        len(seq),
                        tier=tier,
                    )
                    self.block_manager.metrics.record_lookup_time(
                        seq.cpu_lookup_time_ms + timer.elapsed_ms()
                    )
                    seq.metrics_lookup_recorded = True
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break
            if not seq.block_table:
                if use_cpu:
                    self.block_manager.allocate_for_cpu_restore(
                        seq, num_cached_blocks
                    )
                else:
                    self.block_manager.allocate(seq, num_cached_blocks)
                    seq.cache_hit_tier = "gpu" if num_cached_blocks else "miss"
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def confirm_cpu_restore(self, seq: Sequence) -> None:
        restored_blocks = len(seq.cpu_cache_handles)
        self.block_manager.register_restored_blocks(seq, restored_blocks)
        seq.cpu_restore_pending = False

    def rollback_cpu_restore(self, seq: Sequence) -> None:
        matched_blocks = len(seq.cpu_cache_handles)
        matched_tokens = matched_blocks * self.block_size
        if seq in self.running:
            self.running.remove(seq)
        self.block_manager.deallocate(seq)
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        seq.num_scheduled_tokens = 0
        seq.cpu_restore_pending = False
        seq.cpu_restore_failed = True
        seq.cpu_cache_handles = ()
        seq.cache_hit_tier = "miss"
        if seq not in self.waiting:
            self.waiting.appendleft(seq)
        if self.block_manager.metrics is not None:
            self.block_manager.metrics.record_cpu_restore_fallback(
                matched_blocks, matched_tokens
            )

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):
            if not self.sparse:
                self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                if not self.sparse:
                    self.block_manager.deallocate(seq)
                self.running.remove(seq)
