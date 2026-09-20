# Milestone 2: reusable block-store abstraction

Milestone 2 introduces the type and ownership boundary needed by the later CPU
tier. It does not yet persist or transfer KV blocks.

## Ownership boundary

```text
Scheduler / BlockManager
  owns: token blocks, chained hashes, ref-counts, free/used GPU IDs
                         |
                         | physical block ID
                         v
ModelRunner / GPUBlockStore
  owns: [K/V, layer, physical block, token, KV head, head dim] tensor
                         |
                         v
Attention layer
  receives: per-layer K and V views
```

`BlockManager` does not access CUDA tensors. `GPUBlockStore` is constructed only
inside `ModelRunner`, the existing tensor owner. With `enable_reusable_cache=False`
(the default), layer views are assigned exactly as in upstream. With the flag
enabled, the adapter returns the same views; no CPU tier exists yet.

## Identity and lifetime

`CacheFingerprint` includes the resolved model identity, a canonical model-config
digest, dtype, layer count, local KV-head count, head dimension, block size,
tensor-parallel size, RoPE theta, and canonical RoPE scaling. A `CacheKey` pairs
the fingerprint digest with the existing chained token-block hash.

A `GPUBlockHandle` is deliberately documented as ephemeral. When `BlockManager`
reassigns a physical block ID, any view of the old bytes is invalid as context
identity. The `CPUBlockHandle(slot_id, generation)` type is defined now so the
later CPU store can reject stale slot references; no CPU handle is allocated in
this milestone.

`KVBlockPayload` carries both the fingerprint and the physical block tensor.
`GPUBlockStore.write_block()` rejects fingerprint, shape, dtype, and bounds
mismatches before copying.

## Validation

Pure CPU tests validate deterministic fingerprints, layout/dtype checks, block
view ownership, writes, and mismatch failures. The real GPU adapter path is
validated with:

```bash
python benchmarks/validate_instrumentation.py \
  --model /opt/models/Qwen3-0.6B \
  --enable-reusable-cache \
  --output benchmarks/results/gpu_store_validation.json

python benchmarks/deterministic_baseline.py \
  --model /opt/models/Qwen3-0.6B \
  --max-tokens 8 \
  --enable-reusable-cache \
  --output benchmarks/results/gpu_store_equivalence.json
```

The first command reruns all five Milestone 1 prefix-cache cases through the
adapter. The second must produce the same greedy token IDs as the feature-off
baseline. Timing numbers are evidence that the path ran, not a Milestone 6
performance claim.
