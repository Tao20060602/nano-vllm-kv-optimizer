# NanoKV: known limitations

This document lists what NanoKV does **not** do and the configuration under
which it was validated. It is intentionally explicit about missing features so
the project is not mistaken for a production serving system.

## Validated configuration only

NanoKV was developed and benchmarked under a single configuration:

- One GPU, tensor parallel size one;
- eager mode (no CUDA graphs, no torch.compile);
- batch size one / one scheduled sequence at a time;
- block size 256 tokens;
- Qwen3-0.6B in bfloat16;
- WSL2 on an NVIDIA GeForce RTX 3080 Laptop GPU (16 GiB).

Other model families, dtypes, block sizes, TP>1, CUDA graphs, or non-eager
paths are rejected at startup rather than silently misbehaving. The cache
fingerprint includes model name, dtype, number of layers, number of KV heads,
head dimension, block size, and TP size, so a layout mismatch causes a miss
instead of a corrupt restore.

## Synchronous copies only

CPU <-> GPU copies are synchronous on the calling thread. There is no
async copy engine, no overlap of H2D transfer with prefill of the suffix,
and no compute/transfer pipeline. This keeps the implementation easy to read
but caps achievable TTFT gains.

## Process-local CPU memory

The CPU KV store lives in host RAM of the same process. There is no:

- compression or quantization of stored KV;
- NUMA-aware placement;
- swap/SSD tier;
- remote or shared-memory tier;
- cross-process or multi-GPU CPU pool.

Capacity is a fixed number of complete blocks (64 blocks = 16,384 tokens in
the M6 benchmark). LRU eviction reclaims the least-recently-used block but
cannot spill to disk.

## Single-sequence, non-batched path

The scheduler integration forces one scheduled sequence. Chunked prefill,
continuous batching, and multi-sequence decode are not part of the validated
path. The engine does not reject them outright, but they have not been
tested against the CPU store.

## Not implemented (and not claimed)

NanoKV explicitly does **not** implement:

- Sparse attention;
- DIPR or any ANN / vector-index-based lookup;
- production-grade distributed serving;
- a complete AlayaDB replacement;
- speculative decoding;
- prefix caching across processes;
- speculative or async eviction;
- multi-round prefill-chunk interaction with CPU restore.

## Benchmark scope

The M6 benchmark sweeps prefix lengths 256-4096 with `max_model_len=4352`.
An 8192-token prefix is recorded as "not run" rather than fabricated, because
it would exceed the context window and would change the KV cache budget.
Throughput-style benchmarks (continuous batching, decode-heavy workloads) are
not covered; the M6 report measures TTFT and partial-reuse behavior only.

## WSL2-specific notes

Inside WSL2 the host driver is newer than the build-time CUDA toolkit.
Pinned memory worked without memory pressure at 64 blocks (~1.64 GiB), but
this is machine-specific. If a future run causes host memory pressure, reduce
the CPU store capacity rather than treating OOM as a performance result.

## Error recovery

On a CPU restore failure the request falls back to a full prefill of the
prompt and is marked ineligible for another CPU attempt. There is no
automatic retry, no circuit breaker across requests, and no reporting beyond
`get_last_cpu_restore_error()`.