"""NanoKV observability primitives.

The reusable CPU-backed store is introduced in later milestones. This package
currently contains the metrics contract used to observe the upstream GPU cache.
"""

from nanovllm.kvdb.metrics import CacheMetrics

__all__ = ["CacheMetrics"]
