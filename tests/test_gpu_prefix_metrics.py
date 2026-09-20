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


def _lookup_after_materializing(
    first_tokens: list[int], second_tokens: list[int]
) -> tuple[int, dict]:
    metrics = CacheMetrics()
    manager = BlockManager(num_blocks=8, block_size=4, metrics=metrics)
    Sequence.block_size = 4
    first = Sequence(first_tokens)
    _materialize(manager, first)
    metrics.reset()

    second = Sequence(second_tokens)
    cached = manager.can_allocate(second)
    return cached, metrics.snapshot()


def test_cold_miss_is_counted():
    metrics = CacheMetrics()
    manager = BlockManager(num_blocks=8, block_size=4, metrics=metrics)
    Sequence.block_size = 4
    seq = Sequence(list(range(9)))

    assert manager.can_allocate(seq) == 0
    snapshot = metrics.snapshot()
    assert snapshot["requests"] == 1
    assert snapshot["hits"] == 0
    assert snapshot["misses"] == 1
    assert snapshot["reused_tokens"] == 0
    assert snapshot["recomputed_tokens"] == 9


def test_same_prompt_gpu_hit_counts_only_reusable_blocks():
    cached, snapshot = _lookup_after_materializing(list(range(9)), list(range(9)))

    assert cached == 2
    assert snapshot["requests"] == 1
    assert snapshot["hits"] == 1
    assert snapshot["reused_tokens"] == 8
    assert snapshot["recomputed_tokens"] == 1


def test_one_block_partial_prefix_hit():
    cached, snapshot = _lookup_after_materializing(
        list(range(12)),
        [0, 1, 2, 3, 99, 100, 101, 102, 103, 104, 105, 106],
    )

    assert cached == 1
    assert snapshot["gpu_hit_blocks"] == 1
    assert snapshot["reused_tokens"] == 4
    assert snapshot["recomputed_tokens"] == 8


def test_completely_different_prompt_is_a_miss():
    cached, snapshot = _lookup_after_materializing(
        list(range(12)), list(range(100, 112))
    )

    assert cached == 0
    assert snapshot["misses"] == 1
    assert snapshot["reused_tokens"] == 0
    assert snapshot["recomputed_tokens"] == 12


def test_non_aligned_tail_is_not_counted_as_cached():
    cached, snapshot = _lookup_after_materializing(
        list(range(12)),
        [0, 1, 2, 3, 4, 5, 99, 100, 101, 102, 103, 104],
    )

    # Six token IDs are shared, but only the first complete 4-token block is
    # reusable; the two-token tail is included in recomputation.
    assert cached == 1
    assert snapshot["reused_tokens"] == 4
    assert snapshot["recomputed_tokens"] == 8
