from nanovllm.kvdb.metrics import CacheMetrics


def test_metrics_snapshot_is_structured_and_serializable():
    metrics = CacheMetrics()
    metrics.record_lookup(matched_blocks=2, matched_tokens=8, total_tokens=12)
    metrics.record_lookup_time(1.5)
    metrics.record_prefill(3.0)
    metrics.record_ttft(4.0)
    metrics.record_decode(2.0)

    assert metrics.snapshot() == {
        "requests": 1,
        "hits": 1,
        "misses": 0,
        "gpu_hit_blocks": 2,
        "cpu_hit_blocks": 0,
        "reused_tokens": 8,
        "recomputed_tokens": 4,
        "lookup_time_ms": 1.5,
        "load_time_ms": 0.0,
        "store_time_ms": 0.0,
        "prefill_time_ms": 3.0,
        "first_token_time_ms": 4.0,
        "decode_time_ms": 2.0,
        "ttft_ms": 4.0,
        "h2d_bytes": 0,
        "d2h_bytes": 0,
        "eviction_count": 0,
    }


def test_metrics_reset_clears_counters():
    metrics = CacheMetrics()
    metrics.record_lookup(matched_blocks=0, matched_tokens=0, total_tokens=4)
    metrics.reset()
    assert all(value == 0 for value in metrics.snapshot().values())
