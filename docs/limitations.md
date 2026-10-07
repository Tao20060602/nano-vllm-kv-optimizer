# Current limitations and evidence boundary

This document describes the M12–M22 sparse-decode baseline and the bounded
segmented-prefill bridge measured on 2026-10-07. The NanoKV engine optimization
follow-up is deferred and is not complete; its next-stage objective is recorded
in [NATIVE_ENGINE_OPTIMIZATION_NEXT_SCOPE.md](NATIVE_ENGINE_OPTIMIZATION_NEXT_SCOPE.md).
Earlier M0–M6 CPUBlockStore caveats are separated at the end because that
prefix-reuse path and its measurements do not describe the current Qwen3-4B
configuration.

## Reference configuration and workload

The recent experiments use Qwen3-4B BF16 snapshot
`1cfa9a7208912126459214e8b04321603b3df60c` on one RTX 3080 Laptop GPU
(16 GiB), TP=1, eager execution, YaRN, 64-token history blocks, `r=4`
representatives, 4096-token prefill chunks, Top-32 selection, a 64-token sink,
and a 512-token recent window. The primary decode comparisons use one
repeated-text 32K prompt and greedy generation.

The current path keeps full post-RoPE K/V history in CPU memory. GPU memory
holds the selector representatives, sink and recent windows, and the packed
attention buffer. Decode selects historical blocks, gathers their K/V from
CPU into pinned staging, and combines them with the protected GPU-resident
sink/recent context.

The performance workload is a narrow single-sequence engineering workload,
not a diverse-prompt latency or throughput benchmark. M22 separately measures
a bounded RULER-derived quality screen of 80 fixed prompts. Matching generated
IDs and selected-block histories checks tested runs for equivalence; it is not
a quality score. See the [final quality report](m22_quality_closeout_results.md).

M22 finds an explicit quality cost: at 8K, dense/sparse scores are 100/80 for
similar-key distractor retrieval and 97/92 for variable tracking (item recall).
Both sparse arms fail all ten 32K distractor examples; there is no 32K dense
control. M21 matches the sparse baseline's token IDs and answer text on 80/80
prompts, which does not make the underlying sparse approximation lossless.
All 220 generations reach the fixed task cap. No broad 128K quality or
effective-context-length claim is supported.

## Performance experiments and defaults

No performance option was promoted to a default in M19–M21. The current
direct pinned gather default comes from M18; other recent candidates remain
opt-in:

| Path | Default | What the available evidence supports |
| --- | --- | --- |
| Direct pinned gather using `index_select(..., out=pinned)` (M18) | On | Three matched fresh-process 32K pairs had identical generated IDs. Each favored direct gather, with paired steady-median reductions from 11.8% to 18.8% and a 14.7% mean paired reduction on one repeated prompt. This is not a general speedup estimate. |
| Static protected-block mask (M21) | Off | Two fresh-process pairs matched generated IDs and all recorded selected-block histories. Drop4 mean was 7.40% and 10.46% lower per pair on the same prompt; two pairs do not establish a general effect. |
| Selector CUDA graph (M19) | Off | Three corrected fresh-process pairs had mixed timing directions. A faster same-process interleaved run is not enough to change the default. |
| K/V copy pipeline (M20) | Off | The latest ABBA and two fresh-process comparisons showed no acceleration. These observations do not prove the pipeline causes a slowdown. |
| Adaptive decode Top-K or reduced prefill Top-K (M16–M17) | Off | Adaptive decode did not show a TPOT gain. In a small matched multi-key screen, prefill K=32 scored 1/3 while K=24 and K=16 each scored 0/3; lower prefill budgets therefore remain experimental. |
| Segmented GQA prefill bridge (2026-10-07) | Experimental backends opt-in; `flash` remains default | The three-arm suite found no stable end-to-end prefill gain. Operator versus `flash_reuse` geometric mean was 0.999172 (0.08% faster, mixed directions); versus original `flash` it was 1.005987 (0.60% slower across all six pairs). This is not decode or TPOT evidence. |

The M18 gather measurements concern decode on one prompt and do not establish
a prefill speedup. Nsight ranges, shape-derived transfer estimates, and
synthetic gather microbenchmarks are diagnostic; they are not end-to-end
latency or hardware-counter evidence. Detailed methods and evidence are in the
[M16](m16_dynamic_topk_results.md), [M17](m17_prefill_budget_results.md),
[M18](m18_nsight_gather_results.md), [M19](m19_selector_graph_results.md),
[M20](m20_gather_results.md), and
[M21](m21_selector_static_mask_results.md) reports.

The narrow historical decode results remain separate: M18's direct pinned
gather reported a 14.7% mean paired reduction in steady decode median across
three pairs on one repeated 32K prompt; M21's optional static mask reported
7.40% and 10.46% lower Drop4 mean latency in two 32K pairs. These do not predict
prefill performance or generalize beyond those tested workloads.

### Segmented prefill results and partial profiling status (2026-10-07)

The native segmented adapter report covers two fixed 16,480-token prompts,
three fresh-process groups, and only the first generated token. The comparison
supports retaining the existing `flash` default. Attention outputs passed the
existing numeric tolerance, but subsequent selector sets differed in layers
after the 96-token tail chunk. The experiment did not establish model-quality
equivalence or a decode TPOT effect. See
[NATIVE_SEGMENTED_ADAPTER_REPORT.md](NATIVE_SEGMENTED_ADAPTER_REPORT.md) and
the [paired result summary](../benchmarks/results/operator_bridge/m19-native/summary.json).

The registered profiling plan specified all three backends. Following the
priority change, only `flash` and `flash_reuse` were profiled, for one archive
prompt at a 4096-token main chunk and a 96-token tail chunk. The `operator`
profile was not run, so these captures are not a complete three-arm profiling
comparison. The instrumented traces are local ignored artifacts under
`bench_logs/operator_bridge/closeout-profile-20261007/`; they are diagnostics,
not clean timing results. They do not establish that CPU gather, prefill, or
any other stage is the system bottleneck. The engine follow-up is deferred
until the independent operator project is complete; its first step is to
measure the actual critical path, not to assume a gather limitation.

The old NanoKV milestone M19 denotes selector CUDA graph. The segmented
operator bridge is a separate effort and should be referred to by that name
and date, not as NanoKV M19.

## Correctness fixes already incorporated

- M14 fixed the chunked-prefill recent-window gap: the previous recent K/V is
  restored into the attention input before a later chunk runs.
- M16 added a per-new-sequence reset for logical lengths and selector state
  while reusing allocated buffers. The pre-fix sample outputs that inherited
  prior-request state are invalid and excluded from the reported results.

These resolved issues are not open limitations of the documented M21 state.

## Current scope limits

- Only one scheduled sequence, TP=1, eager execution, and the reference model
  configuration above are covered by the recent evidence. Continuous batching
  and multi-sequence decode are outside the validated path.
- The query-guided representative selector and block traversal are an
  educational prototype, not a reproduction of AlayaDB's production
  RoarGraph or a complete AlayaDB implementation.
- Full K/V history uses host RAM and grows with the sequence. The current
  path does not provide compression, quantization, SSD/remote storage, or a
  shared cross-process or multi-GPU cache.
- The available quality screens are small diagnostics, not a full official
  RULER evaluation or broad quality assessment. No general quality,
  production-serving, or workload-wide performance claim is made.

## Historical M0–M6 CPUBlockStore caveats

The points below apply to the earlier CPU-backed prefix-reuse implementation
and M6 benchmark only; they are not the current M12–M21 sparse-decode
configuration.

- That path used Qwen3-0.6B BF16 with 256-token blocks and synchronous
  CPU-to-GPU restore. It did not overlap restoration with suffix prefill.
- Its benchmark used a process-local CPU store capped at 64 blocks
  (16,384 tokens). LRU eviction did not spill to disk.
- The old WSL2 run used about 1.64 GiB of pinned host memory at that capacity.
  That observation is machine-specific and should not be treated as a safe
  capacity recommendation for the current full-history path.
- On that old restore path, a copy failure fell back to full prefill and
  disabled another CPU attempt for that request. It had no automatic retry or
  cross-request circuit breaker.
- M6 latency numbers cover TTFT and partial prefix reuse only. They are
  historical and must not be presented as current sparse-decode results.
