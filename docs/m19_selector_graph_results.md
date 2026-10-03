# M19: sparse selector CUDA graph experiment

Status: local experiment on `codex/m19-selector-cuda-graph`, based on
`adce2a5e17946a9d800186ea2bd086fe82c11080`. Measurements used uncommitted
source; the implementation, tests, and reports are now versioned together.
The selector graph remains opt-in; `selector_cuda_graph=False` is the default.
Fresh-process timing was inconsistent, although one same-process interleaved
run favored graph selection. This is not enough evidence to change the default.

## Setup

- GPU: NVIDIA GeForce RTX 3080 Laptop, 16 GiB, SM 8.6.
- Runtime: NanoVLLM-Ubuntu, CUDA 12.8, PyTorch 2.7.1+cu128, Python 3.12,
  Triton 3.3.1, FlashAttention 2.8.3.post1.
- Model: Qwen3-4B snapshot
  `1cfa9a7208912126459214e8b04321603b3df60c`.
- Workload: repeated-text prompt, BF16, YaRN, 4096-token prefill chunks,
  query-guided M12 sparse selector, Top-32, one query summary,
  `index_select=True`, and greedy generation.

The graph captures only the decode selector's score/reduction/Top-K operations.
The arithmetic and Top-K configuration are unchanged. Selected block IDs and
generated token IDs were recorded for exact comparisons.

The first capture implementation entered PyTorch's `torch.cuda.graph`
convenience context once per layer. Its entry path synchronized CUDA, ran
Python GC, and cleared the allocator cache on every capture. That produced
about 4.7 seconds of measured setup across 36 layers. It was replaced with a
warmed side stream and raw `CUDAGraph.capture_begin()/capture_end()` calls.
The corrected implementation built 36 graphs in about 200–296 ms for 32K.
The earlier wrapper results below are retained as an abandoned implementation
diagnostic and are not mixed with corrected measurements.

## Corrected 32K fresh-process A/B

Each arm used a new process. The listed order alternated to expose drift.
Each 32-token run contains 31 measured decode steps; the steady summaries drop
the first four decode steps. A negative median delta means graph was faster;
a positive delta means graph was slower.

| Pair | Order | Eager median / mean (ms/token) | Graph median / mean (ms/token) | Median delta, graph/eager | Total decode wall, eager / graph (ms) |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | eager → graph | 197.96 / 203.83 | 170.88 / 176.31 | -13.7% | 7584.69 / 6900.56 |
| 2 | graph → eager | 149.12 / 156.39 | 182.16 / 187.92 | +22.2% | 5532.91 / 7612.34 |
| 3 | eager → graph | 132.98 / 135.83 | 136.55 / 144.09 | +2.7% | 5144.11 / 5690.97 |

All three pairs produced identical 32-token outputs and identical selector ID
histories for all 36 layers and 31 decode steps. The direction of the timing
effect is mixed, and the eager medians span 132.98–197.96 ms/token. Do not
present the first pair as a stable fresh-process speedup.

| Pair | First decode step, eager / graph (ms) | Graph setup (ms, 36 layers) | Graph builds | Eager / graph allocated, reserved, peak allocated (GiB) |
| --- | ---: | ---: | ---: | --- |
| 1 | 1443.13 / 1642.48 | 270.14 | 36 | 8.62 / 9.20 / 8.94; 8.87 / 10.66 / 8.94 |
| 2 | 809.04 / 1956.31 | 296.30 | 36 | 8.62 / 9.20 / 8.94; 8.87 / 10.66 / 8.94 |
| 3 | 1032.76 / 1332.38 | 200.05 | 36 | 8.62 / 9.20 / 8.94; 8.87 / 10.66 / 8.94 |

The first decode step includes graph setup. In these runs, graph setup was a
one-time 200–296 ms cost, while steady timing varied much more between fresh
processes. Graph arms ended with about 0.25 GiB more allocated memory and
1.46 GiB more reserved memory; peak allocated memory was unchanged.

## Same-process interleaved control

The benchmark now accepts `--selector-abba` together with
`--selector-cuda-graph`. It warms four decode steps with graphs enabled, then
repeats eager/graph/graph/eager. The option records the mode at each decode
step and the 36-layer selector host time for that step. It rejects profiler and
finite-check options. The normal benchmark path is unchanged when this option
is absent.

One 32K run generated 48 tokens and recorded 47 decode steps. After the four
graph warmups, it had 21 eager and 22 graph samples:

| Arm | Samples | Decode median / mean (ms) | 36-layer selector median / mean (ms) |
| --- | ---: | ---: | ---: |
| Eager | 21 | 132.16 / 137.21 | 24.40 / 26.73 |
| Graph | 22 | 112.21 / 114.83 | 7.41 / 7.54 |

Within this run, graph decode steps were 15.1% faster by median and 16.3%
faster by mean; measured selector host time was 69.6% lower by median and
71.8% lower by mean. The first decode step was 991.55 ms and includes capture
setup. Setup summed to 216.42 ms across 36 graphs. Allocated/reserved/peak GPU
memory after the run was 8.87/10.66/8.94 GiB.

The first 32 generated IDs and the first 31 decode-step selector histories
matched the corrected raw eager baseline exactly. The ABBA arms occupy
different token positions in one deterministic trajectory, so this is a
same-process control, not a same-query paired comparison.

As a simple amortization estimate, 216.42 ms of measured graph setup divided
by the 22.38 ms mean steady decode-step difference gives about 9.7
graph-enabled steady steps to recover setup. This estimate assumes the
single-run per-step difference persists and excludes broader run-to-run
variation; it is not an end-to-end service claim.

## 64K scaling and correctness check

One fresh-process 64K/16-token pair was run. The graph arm left 12.32 GiB host
memory available. Both arms produced identical 16-token outputs and all
36-layer selector ID histories across 15 decode steps.

| Arm | Steady drop-4 median / mean (ms/token) | First decode step (ms) | Total decode wall (ms) | Graph setup |
| --- | ---: | ---: | ---: | ---: |
| Eager | 473.17 / 473.63 | 1616.75 | 8254.64 | — |
| Graph | 471.17 / 492.96 | 1870.61 | 8699.09 | 650.11 ms, 36 graphs |

This short pair is a correctness and scaling guard only: its medians are nearly
equal, its graph mean and total decode wall are higher, and it has only 11
drop-4 steps.

## Nsight diagnostics

Two 32K/8-token step-4 captures are preserved:

- `m19_decode_graph_32k_step4.nsys-rep` used
  `--cuda-graph-trace=node`. Node tracing substantially perturbed the
  selector and transfer ranges; its wall and stage totals are not usable as
  bottleneck shares.
- `m19_decode_graph_light_32k_step4.nsys-rep` used
  `--cuda-graph-trace=graph` and is the preferred GUI trace.

The light trace's CUDA API summary for the captured step reports 434
`cudaLaunchKernel` calls, 511 `cudaMemcpyAsync` calls, and 36
`cudaGraphLaunch` calls. The comparable M18 index-select trace reports 974
`cudaLaunchKernel` calls and 583 `cudaMemcpyAsync` calls. This shows the
API structure: the M19 path submits 36 selector graphs instead of hundreds of
individual kernel-launch API calls. These counts are not a kernel count or a
latency comparison; graph internals are represented at graph granularity.

NVTX host summaries for the M18 index-select capture were 37.72 ms selector,
3.20 ms nested selector ID-to-host, and 33.17 ms CPU gather across 36 layers.
The M19 graph-granularity trace reported 107.80 ms, 89.77 ms, and 128.30 ms
for those ranges; the nested ID-to-host value is already inside selector time.
The node-granularity trace reported 221.60 ms selector, 212.07 ms nested
ID-to-host, and 37.52 ms CPU gather. These profiled host ranges are strongly
perturbed and are not compared as speed measurements. CUDA H2D transfers are
asynchronous, so NVTX host ranges and device copy durations may overlap and
must not be added as an exclusive wall-time breakdown. The traces do not
establish a bottleneck share or a causal explanation for the timing changes.

## Earlier convenience-context runs

Before replacing the capture API, two 32K pairs had exact output and selector
ID matches but approximately 4.68 seconds of graph setup across 36 layers:

| Pair | Eager median / mean | Graph median / mean | First decode, eager / graph | Graph setup |
| --- | ---: | ---: | ---: | ---: |
| 1 | 131.55 / 133.82 ms | 146.66 / 157.19 ms | 773.53 / 5752.30 ms | 4684.92 ms |
| 2 | 131.93 / 136.09 ms | 109.95 / 112.65 ms | 750.33 / 5537.25 ms | 4677.65 ms |

These artifacts document the costly capture wrapper only:
`m19_decode_ab_1_0/1_1.json` and `m19_decode_ab_2_0/2_1.json`, with matching
`.log` files. The separate `m19_decode_ab_.json/.log` files came from a
malformed PowerShell loop command and were excluded from all results.

## Scope and conclusion

All correctness checks here use one repeated-text prompt and greedy decoding.
They establish exact selected IDs and token outputs for the tested runs, not
long-context quality, diverse prompts, or production serving behavior.

The interleaved run supports an opt-in selector graph improvement for this
one 32K trajectory, but three corrected fresh-process pairs disagree on
direction. Keep the runtime default off until the steady effect repeats across
fresh processes or representative prompts. The Nsight captures explain launch
structure only; they do not replace unprofiled end-to-end timing.

## Reproduction

Run inside `NanoVLLM-Ubuntu` from `/opt/nano-vllm`:

```bash
source .venv/bin/activate
export HF_HOME=/opt/models/.cache/huggingface

# Corrected 32K fresh-process pairs. Run each line in its own fresh process.
python benchmarks/benchmark_m15_decode.py --seq-len 32768 --gen-tokens 32 --index-select 1 --record-selected-ids --output bench_logs/m19_decode_raw_ab_1_0.json > bench_logs/m19_decode_raw_ab_1_0.log 2>&1
python benchmarks/benchmark_m15_decode.py --seq-len 32768 --gen-tokens 32 --index-select 1 --record-selected-ids --selector-cuda-graph --output bench_logs/m19_decode_raw_ab_1_1.json > bench_logs/m19_decode_raw_ab_1_1.log 2>&1
python benchmarks/benchmark_m15_decode.py --seq-len 32768 --gen-tokens 32 --index-select 1 --record-selected-ids --selector-cuda-graph --output bench_logs/m19_decode_raw_ab_2_1.json > bench_logs/m19_decode_raw_ab_2_1.log 2>&1
python benchmarks/benchmark_m15_decode.py --seq-len 32768 --gen-tokens 32 --index-select 1 --record-selected-ids --output bench_logs/m19_decode_raw_ab_2_0.json > bench_logs/m19_decode_raw_ab_2_0.log 2>&1
python benchmarks/benchmark_m15_decode.py --seq-len 32768 --gen-tokens 32 --index-select 1 --record-selected-ids --output bench_logs/m19_decode_raw_ab_3_0.json > bench_logs/m19_decode_raw_ab_3_0.log 2>&1
python benchmarks/benchmark_m15_decode.py --seq-len 32768 --gen-tokens 32 --index-select 1 --record-selected-ids --selector-cuda-graph --output bench_logs/m19_decode_raw_ab_3_1.json > bench_logs/m19_decode_raw_ab_3_1.log 2>&1

# Same-process interleaved control.
python benchmarks/benchmark_m15_decode.py --seq-len 32768 --gen-tokens 48 --index-select 1 --selector-cuda-graph --selector-abba --record-selected-ids --output bench_logs/m19_decode_interleaved_32k.json > bench_logs/m19_decode_interleaved_32k.log 2>&1

# Single 64K correctness/scaling pair.
python benchmarks/benchmark_m15_decode.py --seq-len 65536 --gen-tokens 16 --index-select 1 --record-selected-ids --output bench_logs/m19_decode_64k_0.json > bench_logs/m19_decode_64k_0.log 2>&1
python benchmarks/benchmark_m15_decode.py --seq-len 65536 --gen-tokens 16 --index-select 1 --record-selected-ids --selector-cuda-graph --output bench_logs/m19_decode_64k_1.json > bench_logs/m19_decode_64k_1.log 2>&1

# Graph-granularity Nsight trace; diagnostic only.
nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --cuda-graph-trace=graph -o bench_logs/m19_decode_graph_light_32k_step4 \
  python benchmarks/benchmark_m15_decode.py --seq-len 32768 --gen-tokens 8 \
    --index-select 1 --selector-cuda-graph --nsys-trace-step 4 \
    --output bench_logs/m19_decode_graph_light_32k_step4.json
nsys stats --force-export=true --report cuda_api_sum,nvtx_sum \
  bench_logs/m19_decode_graph_light_32k_step4.nsys-rep
```

The default model is the pinned Qwen3-4B snapshot above. Benchmark JSON,
logs, and Nsight reports are local `bench_logs/` artifacts.

## Source identifiers

Tracked base: `adce2a5e17946a9d800186ea2bd086fe82c11080`.
SHA-256 at measurement time:

| File | SHA-256 |
| --- | --- |
| `benchmarks/benchmark_m15_decode.py` | `5120b7e08cd466c2ad68903b4f901ac5a7d266855b7493f4d291a745c3810d21` |
| `nanovllm/config.py` | `9ed8bd614bfc39a0e8d8f96c69d8bb4bea56d136b4a3df15e98c1fb49f13e225` |
| `nanovllm/engine/model_runner.py` | `b27ff21f0abe3c760c6f870e321cb134acb4b582b206fadcb860cd4301b71630` |
| `nanovllm/sparse/m12_runtime.py` | `6befe29b607a14a14c4a6be552faedf6902d0417fac1b9d36f5343c2d7e28e2f` |
| `tests/test_m19_selector_graph.py` | `7594667df2d09dbc8aad78b9e6787e2b1b0bf45823cf75e718d970edd1cd866` |

Final verification after the raw-capture change passed all 29 targeted tests
in `test_m14_prefill_layout.py`, `test_m16_dynamic_topk.py`,
`test_m17_prefill_budget.py`, and `test_m19_selector_graph.py`, including
10 selector-graph tests. The benchmark, runtime, config, and model runner
passed `python -m py_compile`; `git diff --check` was clean.
No commit or push had been made at measurement time. The closeout includes
the implementation, tests, and this report in version control.

The preferred light Nsight report was copied, with matching SHA-256, to
`C:\Users\28898\Documents\ChatGPT\nano-vllm找实习\NsightReports\M19\m19_decode_graph_light_32k_step4.nsys-rep`
for the native Windows GUI. The node-granularity report remains a local
diagnostic artifact, not performance evidence.
