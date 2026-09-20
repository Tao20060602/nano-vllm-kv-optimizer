from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence
from nanovllm.kvdb.metrics import CacheMetrics


def _materialize(manager: BlockManager, seq: Sequence) -> None:
    cached = manager.can_allocate(seq)
    assert cached >= 0
    manager.allocate(seq, cached)
    seq.num_scheduled_tokens = len(seq)
    manager.hash_blocks(seq)
    seq.num_cached_tokens = len(seq)


def test_cold_miss_and_gpu_hit_are_counted_by_token_block():
    metrics = CacheMetrics()
    manager = BlockManager(num_blocks=8, block_size=4, metrics=metrics)
    Sequence.block_size = 4
    first = Sequence(list(range(8)))
    _materialize(manager, first)

    second = Sequence(list(range(8)) + [99, 100, 101, 102])
    cached = manager.can_allocate(second)

    assert cached == 2
    snapshot = metrics.snapshot()
    assert snapshot["requests"] == 2
    assert snapshot["misses"] == 1
    assert snapshot["hits"] == 1
    assert snapshot["reused_tokens"] == 8
    # The first cold request recomputes 8 tokens; the second recomputes its
    # 4-token suffix, so the cumulative counter is 12.
    assert snapshot["recomputed_tokens"] == 12


def test_non_aligned_tail_is_not_counted_as_cached():
    metrics = CacheMetrics()
    manager = BlockManager(num_blocks=8, block_size=4, metrics=metrics)
    Sequence.block_size = 4
    first = Sequence([1, 2, 3, 4, 5, 6, 7, 8])
    _materialize(manager, first)

    second = Sequence([1, 2, 3, 4, 5, 6, 7, 8, 9])
    assert manager.can_allocate(second) == 2
    assert metrics.snapshot()["reused_tokens"] == 8
