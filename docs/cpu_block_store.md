# Milestone 3: standalone CPUBlockStore

This milestone implements CPU KV storage without connecting it to the scheduler
or prefix index.

## Layout and capacity

Each slot stores one complete physical KV block with shape:

```text
[2 (K/V), layers, block_size, local KV heads, head_dim]
```

The dtype and dimensions come exclusively from `CacheFingerprint`. Capacity is
configured in bytes and rounded down to a whole number of blocks. Construction
fails if even one block cannot fit. `resident_bytes` is therefore always bounded
by the effective `capacity_bytes`.

## Handle safety and LRU

`CPUBlockHandle` contains `(slot_id, generation)`. Every assignment increments a
slot's generation. Loading a deleted or evicted generation raises
`StaleCPUBlockHandle`; an old handle can never silently read newly assigned data.

The LRU order is deterministic. Store, CPU read, and load-to-destination all make
a slot most recently used. When full, the least recently used slot is invalidated
before reuse. Delete and eviction both call `on_invalidate(handle, key)`, which is
the hook Milestone 4 uses to remove PrefixIndex references.

## Transfers and metrics

`store_block()` performs a synchronous copy into pageable or pinned CPU memory.
`load_into()` synchronously copies into the caller-provided destination. CUDA
sources/destinations are synchronized before and after timing. Stats expose
D2H/H2D bytes, total store/load milliseconds, derived GB/s, capacity/residency,
store/load/delete counts, and eviction count.

No compression or dtype conversion occurs, so round trips require exact tensor
equality.

## Validation

```bash
python -m pytest -q tests/test_cpu_block_store.py

python benchmarks/validate_cpu_block_store.py \
  --model /opt/models/Qwen3-0.6B \
  --output benchmarks/results/cpu_block_store_validation.json
```

Unit tests cover multiple layer/K/V layouts and dtypes, pageable and pinned
allocation, LRU access order, over-capacity construction, delete/reinsert,
fingerprint/shape/dtype mismatch, invalidation callbacks, stale handles, and 100
store/load/evict cycles. The GPU validation copies one real Qwen3-0.6B-size block
(`28 MiB`) through both pageable and pinned stores and asserts exact equality.
