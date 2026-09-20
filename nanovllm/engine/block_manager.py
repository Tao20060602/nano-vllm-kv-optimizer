from collections import deque
from nanovllm.engine.sequence import Sequence
from nanovllm.kvdb.metrics import StageTimer
from nanovllm.kvdb.prefix_index import compute_block_hash


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int, metrics=None):
        self.block_size = block_size
        self.metrics = metrics
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        return compute_block_hash(token_ids, prefix)

    def _allocate_block(self) -> int:
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence, *, record_metrics: bool = True) -> int:
        timer = StageTimer() if self.metrics is not None and record_metrics else None
        if timer is not None:
            timer.__enter__()
        h = -1
        num_cached_blocks = 0
        num_new_blocks = seq.num_blocks
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break
            num_cached_blocks += 1
            if block_id in self.used_block_ids:
                num_new_blocks -= 1
        if self.metrics is not None and record_metrics and not seq.metrics_lookup_recorded:
            self.metrics.record_lookup(num_cached_blocks, num_cached_blocks * self.block_size, len(seq))
            self.metrics.record_lookup_time(timer.elapsed_ms())
            seq.metrics_lookup_recorded = True
        if len(self.free_block_ids) < num_new_blocks:
            return -1
        return num_cached_blocks

    def can_allocate_fresh(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        assert not seq.block_table
        h = -1
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                block.ref_count += 1
            else:
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
        for i in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    def allocate_for_cpu_restore(self, seq: Sequence, num_cached_blocks: int):
        assert not seq.block_table
        assert self.can_allocate_fresh(seq)
        for _ in range(seq.num_blocks):
            seq.block_table.append(self._allocate_block())
        seq.num_cached_tokens = num_cached_blocks * self.block_size
        seq.cpu_restore_pending = True
        seq.cache_hit_tier = "cpu"

    def register_restored_blocks(self, seq: Sequence, num_restored_blocks: int):
        prefix_hash = -1
        for i in range(num_restored_blocks):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            prefix_hash = self.compute_hash(token_ids, prefix_hash)
            block.update(prefix_hash, token_ids)
            self.hash_to_block_id[prefix_hash] = block.block_id

    def evict_free_cached_blocks(self) -> int:
        evicted = 0
        for block_id in self.free_block_ids:
            block = self.blocks[block_id]
            if block.hash == -1:
                continue
            if self.hash_to_block_id.get(block.hash) == block_id:
                del self.hash_to_block_id[block.hash]
            block.hash = -1
            block.token_ids = []
            evicted += 1
        return evicted

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())

    def hash_blocks(self, seq: Sequence):
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end: return
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id
