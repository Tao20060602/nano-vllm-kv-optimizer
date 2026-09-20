# NanoKV: correctness evidence

This document collects the correctness guarantees exercised by the test suite
and the integration benchmark. Every claim below maps to an automated test or
a recorded JSON artifact; nothing here is hand-waved.

## Unit and integration tests

Run from `/opt/nano-vllm` in the `NanoVLLM-Ubuntu` distribution:

```bash
python -m pytest -q
```

Result: **28 tests passed** at the M6 commit. The tests cover:

- `tests/test_cache_fingerprint.py` - fingerprint includes model name, dtype,
  layers, KV heads, head dim, block size, and TP size; a mismatch rejects reuse.
- `tests/test_prefix_index.py` - longest-prefix matching, collision safety
  (hash matches but token IDs differ -> miss), and non-aligned tail handling.
- `tests/test_cpu_block_store.py` - CPUBlockStore put/load/evict, capacity
  limit, LRU eviction order, pinned vs pageable paths.
- `tests/test_gpu_block_store.py` - GPUBlockStore block views, reference
  counts, and free-list accounting.
- `tests/test_cache_metrics.py` - metrics reset per `generate()` call,
  lookup/load/prefill/TTFT timing boundaries, zero counters on feature off.
- `tests/test_gpu_prefix_metrics.py` - GPU-only prefix-hit counters.

## End-to-end token-identity evidence (Milestone 5)

`benchmarks/validate_cpu_prefix_reuse.py` compares generated greedy token IDs
against an independently cold prompt for ten cases. Raw results:

- `benchmarks/results/cpu_prefix_baseline.json`
- `benchmarks/results/cpu_prefix_integration.json`

| Case | Scenario | Reused | Actual prefill | Token IDs equal |
| --- | --- | ---: | ---: | --- |
| A | cold miss | 0 | 600 | yes |
| B | exact repeat, GPU hit | 512 | 88 | yes |
| C | shared prefix, GPU hit | 256 | 344 | yes |
| D | GPU identity cleared, CPU hit | 512 | 88 | yes |
| E | CPU partial hit | 256 | 344 | yes |
| F | 300-token non-aligned prefix | 256 | 344 | yes |
| G | first token differs | 0 | 600 | yes |
| H | differs after 2 blocks | 512 | 388 | yes |
| I | CPU LRU eviction | 0 | 600 | yes |
| J | feature disabled | 0 | full prompt | equals upstream |

Case D records exactly 58,720,256 H2D bytes (two 28 MiB blocks) after three
GPU cache identities were explicitly removed.

## Specific guarantees

- **Longest-prefix lookup**: the index walks chained xxHash and stops at the
  first mismatch; it returns the longest block-aligned match, not a first-hit.
- **Collision safety**: after a hash match the stored token IDs are compared
  before a block is reused; a hash collision with different tokens is a miss.
- **Non-aligned tail**: reuse is capped at `floor((prompt_tokens - 1) /
  block_size)` complete blocks; a non-aligned tail is always prefetched.
- **GPU eviction -> CPU restore**: after `clear_gpu_prefix_cache()`, the same
  prompt restores from the CPU store and produces identical tokens.
- **CPU round-trip equivalence**: K/V bytes stored to CPU and loaded back to
  GPU are used directly by FlashAttention; token IDs match the cold baseline.
- **Refcount hygiene**: after every request the GPU allocator returns to
  `used=0` and all physical block refcounts are zero. The LRU/eviction tests
  and the integration case D assert this.
- **Restore failure is transactional**: if any H2D copy throws, freshly
  allocated GPU blocks are deallocated, restore metadata is cleared, the
  request is marked ineligible for another CPU attempt, metrics move from
  "CPU hit" to "recompute miss", and the full prompt is rescheduled. The
  integration test injects this failure and proves the fallback generates
  cold-baseline tokens.
- **Fingerprint gating**: a model, dtype, or KV layout mismatch changes the
  cache fingerprint and therefore never reuses a block from a previous
  configuration.

## Benchmark self-checks (Milestone 6)

`benchmarks/benchmark_prefix_reuse.py` asserts on every measured iteration:

- `metrics.reused_tokens` equals the expected reused count;
- `metrics.prefill_executed_tokens` equals `prompt_tokens - expected_reused`;
- for the partial-reuse mode, `metrics.cpu_hit_blocks` equals the expected
  shared block count.

A failure raises and stops the run rather than being recorded as a silent
"hit". Raw per-iteration records (including `output_token_ids`) are retained
in `benchmarks/results/m6/*.json`.

## Feature flag off = upstream behavior

With `enable_cpu_cache=False` (default), no CPU store, no extra lookup timer,
and no transfer code path are created. The engine follows the upstream
request call chain documented in `docs/architecture.md`. Case J above is the
explicit check that this path produces the same output as the untouched
upstream snapshot.