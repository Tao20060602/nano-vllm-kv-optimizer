# M0-M14 Benchmark Backfill

Date: 2026-09-23.  Authoritative environment: `NanoVLLM-Ubuntu`, RTX 3080
Laptop 16 GiB, PyTorch 2.7.1+cu128, FlashAttention 2.8.3.post1.  Source baseline
was clean `main` commit `288e2b3`; benchmark-harness changes were made on
`codex/benchmark-backfill`.

## Measurement repairs

- Engine steps are enclosed by `torch.cuda.synchronize()` and prefill is
  separated from standalone decode steps.
- `max_tokens=1` is correctly described as sampling the first token in the
  final prefill step; it does not create a separate decode step.
- Matched A/B runs keep model, prompt, chunk, r=4, top-32, sink/recent, YaRN,
  dtype and sampling fixed.
- Decode GPU sub-stages have an opt-in CUDA-event probe; CPU gather remains a
  CPU wall measurement.  Selector already includes its ID D2H synchronization.
- Tensor payload is shape-derived, not a PCIe hardware counter.

## M14 prefill: Torch reference versus FlashAttention-2

Production routing (`query_segments=1`), chunk=4096.  Three alternating runs
per backend; tables report the median of the three run totals.

| Prompt | Torch prefill | FA2 prefill | Reduction | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 8K | 8162.94 ms | 3826.35 ms | **53.1%** | 2.13x |
| 32K | 42003.26 ms | 12718.44 ms | **69.7%** | 3.30x |

At 8K the later 4096-token chunk median fell from 5881.72 to 1612.71 ms
(72.6%).  At 32K, the seven later chunks fell from a 39774.84 ms median total
to 10515.31 ms (73.6%).  The common first dense chunk is not attributed to the
backend optimization.  Peak allocated memory was about 10.18/10.25 GiB for
Torch versus 8.46/8.53 GiB for FA2 at 8K/32K.

## M13 gather: matched sparse decode

32K prompt, 32 greedy tokens, corrected M14 prefill, three alternating runs.

| Path | Median of run steady medians | Last-step CPU gather median |
| --- | ---: | ---: |
| advanced indexing + pageable temporary | 160.69 ms/token | 34.39 ms |
| `index_select(out=pinned)` | **141.84 ms/token** | **24.15 ms** |

The stable reduction is **11.7%** in steady decode median and **29.8%** in the
CPU gather snapshot.  All six runs produced one identical 32-token sequence.
Selected H2D tensor payload remained 288 MiB/token.  A 64K confirmation pair
was 164.44 -> 153.11 ms/token (6.9% lower) and gather 35.09 -> 28.60 ms.

For one profiled 32K index-select run, CUDA-event sums across 36 layers on the
last decode step were selector 22.87 ms, recent 6.75 ms, H2D pack 29.28 ms and
packed attention 6.96 ms; CPU gather wall was 26.10 ms.  The old asynchronous
CPU timer reported H2D as only 8.83 ms in that same snapshot, directly
confirming that it was not a valid GPU-stage duration.

## Matched dense versus sparse boundary

One matched run per mode; Qwen3-4B, YaRN, 16 greedy tokens.  Token sequences
were identical between dense and sparse at both successful lengths.

| Prompt | Dense prefill / decode | Sparse prefill / decode | Peak allocated |
| ---: | ---: | ---: | ---: |
| 8K | 3134.83 ms / 37.40 ms | 3808.76 ms / 138.31 ms | 13.22 vs 8.42 GiB |
| 16K | 6403.99 ms / 35.85 ms | 6592.25 ms / 140.50 ms | 13.19 vs 8.46 GiB |
| 32K | capacity failure | 12420.10 ms / 147.92 ms | no dense result vs 8.53 GiB |

Decode values are steady medians.  The current sparse system is therefore not
a low-latency replacement when dense fits at 8K/16K; it trades latency for a
much lower GPU-memory footprint and continued operation at 32K.  Dense 32K
failed in the paged-KV scheduler because it could not allocate enough GPU
blocks; this was not reported as a CUDA OOM and no fabricated timing is used.

## Long-context quality after the recent-window repair

All prompts assert that both needle and final question token sequences are
present.  Qwen3-4B, 64K, q=1, r=4, top-32, chunk=4096:

- simple needle at 10%: pass;
- simple needle at 50%: pass;
- simple needle at 90%: fail (model emitted filler);
- same-field distractor at 50%: fail (model emitted filler);
- multikey at 50%: the 16-token run truncated before the answer; a controlled
  32-token rerun passed.

The adjusted result is therefore **3/5**, not the old M12 4/5.  q=4 did not
recover either true failure (0/2) and still passed the 32-token multikey case,
so it remains opt-in with no demonstrated quality gain.  A first 128K quality
point (simple needle at 50%) passed in 56.58 s: **1/1 single-case evidence**, not
a general 128K quality claim.

## M0-M7 prefix reuse: current-code confirmation

Qwen3-0.6B, block size 256, five warmups and twenty recorded measurements per
point.  Every exact-reuse mode generated the same token sequence as cold at
all six points.  The table reports median TTFT; parenthesized values are signed
changes relative to cold at the same prompt, so negative is faster.

| Reused prefix | Cold | GPU hit | CPU pageable | CPU pinned |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 44.79 ms | 32.16 ms (-28.2%) | 35.90 ms (-19.9%) | 43.29 ms (-3.3%) |
| 512 | 31.51 ms | 33.44 ms (+6.1%) | 41.72 ms (+32.4%) | 39.04 ms (+23.9%) |
| 1024 | 42.22 ms | 32.25 ms (-23.6%) | 47.79 ms (+13.2%) | 56.90 ms (+34.8%) |
| 2048 | 83.21 ms | 34.08 ms (-59.0%) | 66.33 ms (-20.3%) | 55.81 ms (-32.9%) |
| 4096 | 181.58 ms | **32.58 ms (-82.1%)** | 97.60 ms (-46.2%) | **76.67 ms (-57.8%)** |

The measured CPU-cache crossover is therefore around 2K tokens in this
environment; results at 1K and below are not consistently positive.  The
short-point cold medians are also non-monotonic, so these points should be
treated as fixed-overhead/noise dominated rather than used to claim a smooth
scaling curve.  At 4K, pinned memory saved another 20.93 ms over pageable.

The cumulative cache telemetry at the end of the run reported effective H2D
bandwidth of 10.93 GB/s for pinned versus 7.44 GB/s for pageable; D2H was 7.06
versus 1.70 GB/s.  The configured CPU cache held up to 64 blocks, 1.75 GiB.
These are application-level bytes/time counters, not PCIe hardware counters.

For the 1024-prefix request with 32 generated tokens, end-to-end wall medians
were cold 869.09 ms, GPU 871.22 ms, pageable 872.58 ms and pinned 901.68 ms.
Prefix reuse improved GPU TTFT from 50.23 to 33.36 ms, but the unchanged decode
work dominated the short request, so there is no end-to-end speedup claim.

Partial pinned reuse used a fixed 2064-token prompt.  Relative to the 83.21 ms
cold reference, median TTFT was 91.20, 72.80, 54.95 and 54.81 ms for 25%, 50%,
75% and 100% reusable full blocks.  The 75% and 100% points being effectively
tied shows the local tradeoff: transferring additional old KV can cost about
as much as recomputing the remaining small suffix.

## M8-M11 historical comparisons rerun

- **M8, 8192 random tensors:** exact Block-DIPR retained critical-token recall
  1.0 while selecting about 52% of tokens in this synthetic union-head setup;
  exact selection plus sparse attention remained slower than dense attention,
  confirming its oracle role rather than a speed claim.
- **M9, Qwen3-0.6B real post-RoPE trace:** beta=48 Route A selected about 75%
  of tokens, recovered about 98% attention mass, critical-token recall 1.0 and
  relative L2 error about 0.02.  CPU search alone was about 3.97 ms median, so
  synchronous offload did not beat GPU-resident attention.
- **M10, 8K real trace:** flat r=4 representative search was roughly 1-2 ms;
  Python KNN/query-guided graph search was roughly 24-53 ms.  At the comparable
  degree-16/visited-64 beta-80 point, query-guided graph recall 0.609 exceeded
  KNN 0.500, but search remained about 43 ms.  This supports the documented
  choice of flat GPU-friendly retrieval without claiming graph retrieval is
  universally ineffective.
- **M11, repaired 2K prompt:** the final question is now guaranteed present.
  Dense answered 74291 at 26.4 ms decode p50.  Flat-real, KNN graph and
  query-guided graph all answered correctly; their p50 values were about
  437.2, 906.2 and 909.9 ms respectively.  Graph paths were about 2.1x slower
  than flat-real at the same observed answer outcome.

## Regression and remaining limits

Full repository regression: **91 passed in 674.71 s**.  M14 targeted CUDA
layout/backend tests: **6 passed**.

Still not established:

- multi-run 64K/128K TTFT stability;
- a dense 32K latency baseline (capacity gate prevents execution here);
- broad 128K quality (only one case);
- RULER/NeedleBench or production workloads;
- decode continuations beyond the 512-token recent window;
- hardware-counter PCIe bandwidth;
- per-Q-block (`Q block=64`) retrieval, which is a proposed next experiment,
  not part of this backfill.

All tracked raw records use the `benchmarks/results/backfill_*` prefix.  `.log`
files remain local diagnostic output; JSON/CSV are the portable evidence.
