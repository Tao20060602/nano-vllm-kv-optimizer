# M21: decode static protected-block mask

Status: decode-only `selector_static_mask` is opt-in and defaults to `False`;
the M20 K/V pipeline remains opt-in and default-off. Measurements used
uncommitted source on `codex/m19-selector-cuda-graph`, based on
`adce2a5e17946a9d800186ea2bd086fe82c11080`. The implementation, tests, and
reports are now versioned together; JSON `git_head` identifies the measurement
baseline, not the complete experimental source.

## Change

The selector previously rebuilt protected-block tensors on the GPU, used GPU boolean filtering and
advanced-index assignment from a host scalar. M21 filters the protected-block tuple on CPU, caches
valid GPU indices, and applies `index_fill_` during decode. Tests cover cache reset, protected-set
updates, and history invalidation. Scoring, Top-K, prefill, CUDA graph selection, and packed attention
are unchanged.

The M21 serial/pipeline step-4 traces showed 540 `cudaLaunchKernel`, 108 `cudaMemcpyAsync`, and 108
`cudaStreamSynchronize` calls in each selector pre-ID range. The inner selector ID-to-host range had
36 copies and 36 waits. Static mask removed the pre-ID copies and waits while retaining the ID-to-host
copies and necessary synchronization.

## Workload and correctness

All four unprofiled arms used pinned Qwen3-4B snapshot `1cfa9a7208912126459214e8b04321603b3df60c`,
32K repeated-text input, greedy 32-token generation, Top-32, 36 layers, eight CPU threads, and
`index_select=True`. K/V pipeline, selector CUDA graph, and finite checks were off; selected IDs were
recorded. Pair 1 ran serial then static mask; Pair 2 reversed the order. Both pairs produced identical
32-token outputs and 36×31 selected-block ID histories. The Nsight serial/static-mask captures also
matched on eight generated IDs and 36×7 captured histories. These checks compare IDs and outputs,
not score values.

## Unprofiled fresh-process pairs

Full mean/median use all 31 decode steps; Drop4 excludes the first four. The first step includes setup
costs; decode totals sum all 31 steps. These are unprofiled per-run observations.

| Pair order | Arm | Full mean / median (ms) | Drop4 mean / median (ms) | First step (ms) | Decode total (ms) |
|---|---|---:|---:|---:|---:|
| 1: serial → mask | Serial | 147.973 / 128.717 | 128.265 / 128.203 | 721.977 | 4,587.166 |
|  | Static mask | 135.842 / 118.125 | 118.769 / 118.125 | 655.838 | 4,211.093 |
| 2: mask → serial | Static mask | 135.212 / 115.769 | 117.537 / 115.769 | 662.150 | 4,191.559 |
|  | Serial | 149.722 / 130.309 | 131.264 / 129.456 | 678.839 | 4,641.384 |

Both pairs show lower measured full and Drop4 summaries for static mask. Paired Drop4 mean/median
differences are −7.40%/−7.86% in Pair 1 and −10.46%/−10.57% in Pair 2. These describe two fresh
pairs on one prompt; they are not a general speedup estimate or a production claim.

GPU allocated memory was 8.62037754 GiB for serial and 8.62039471 GiB for static mask in both pairs:
an 18 KiB increase (36 × 512-byte allocator granules). Reserved (9.19726563 GiB) and peak allocated
(8.94075251 GiB) were unchanged. Across runs, post-run host RSS ranged 7.327–7.391 GiB and available
memory 16.913–17.026 GiB.

## Nsight diagnostics

The serial/pipeline pair used the pinned snapshot, 32K input, eight generated IDs, eight CPU threads,
Top-32, and captured `nanokv.decode.step` 4; only the K/V pipeline flag differed. Static mask is
compared with the serial trace. SQLite found one non-null device `globalPid` per trace; M21 exports
contain no MEMSET table. Coverage below is this process's CUPTI activity union inside the host NVTX
window, not full-GPU utilization.

| Trace | Window (ms) | Kernel union (ms) | Copy union (ms) | Device union (ms) | Uncovered (ms) | Process activity coverage | H2D bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
| Serial | 182.441 | 30.181 | 39.351 | 69.532 | 112.909 | 38.11% | 301,992,652 |
| K/V pipeline | 195.351 | 30.581 | 31.615 | 62.196 | 133.155 | 31.84% | 301,992,652 |
| Static mask | 149.222 | 29.684 | 25.868 | 55.553 | 93.669 | 37.23% | 301,989,916 |

Static-mask versus serial API counts in the captured step were:

| API | Serial | Static mask |
|---|---:|---:|
| `cudaStreamSynchronize` | 146 | 38 |
| `cudaMemcpyAsync` | 583 | 475 |
| `cudaLaunchKernel` | 974 | 794 |
| `cuLaunchKernel` | 363 | 363 |

Within the selector ranges, pre-ID activity changed from 540 launches, 108
copies, and 108 synchronizations to 360 launches, zero copies, and zero
synchronizations. The ID-to-host range remained at 36 copies and 36 waits.
H2D payload changed from 301,992,652 to 301,989,916 bytes: only 2,736 bytes
of metadata difference; the 288 MiB KV payload was unchanged.

Nsight perturbs host and device timing. Window, copy-union, and coverage differences do not establish
an end-to-end speedup or causal benefit. Coverage is not hardware utilization; host NVTX ranges may
overlap or nest; labels intersecting device gaps show temporal correlation only. Do not add stage
sums into a wall-clock breakdown.

## Artifacts and verification

- A/B JSON: `bench_logs/m21_mask_ab_{1,2}_{0,1}.json`; selected IDs and timing samples are retained.
- Serial/pipeline trace analysis: `bench_logs/m21_trace_pair.json`.
- Static-mask trace analysis: `bench_logs/m21_static_mask_trace_pair.json`.
- SQLite: `bench_logs/m21_serial_32k_step4.sqlite`, `m21_pipeline_32k_step4.sqlite`, and
  `m21_static_mask_32k_step4.sqlite`. The three `.nsys-rep` files were copied to `NsightReports/M21/`
  and SHA-256 checked against their source files.
- The main task reports 50 targeted tests passed; Python compilation and `git diff --check` passed.
  Report preparation did not launch additional GPU workloads.

Source SHA-256 values at unprofiled A/B and static-mask trace time (serial/pipeline traces were captured earlier):

| File | SHA-256 |
|---|---|
| `benchmarks/benchmark_m15_decode.py` | `ca565305095c09d7eb65c7dba24a437ec4246d726b0e1111b1f119edf679385e` |
| `nanovllm/config.py` | `812f41bb6c68a5ccad58e74f8e0c0d2228d90e3e3cb0458421aacd374bcc6349` |
| `nanovllm/engine/model_runner.py` | `1f7be547b93639f7893a1a468c7762bbbf0da6a9a202edda56e9d99da4c4ae0e` |
| `nanovllm/sparse/m12_runtime.py` | `7509be43c51eb0534b80ea8efefa939f5375dfdba0361536645af54d5a07e6ad` |

## Reproduction with fresh output names

Run from the Windows project directory. These commands use a new `m21_staticmask_repro_` prefix.

```powershell
.\scripts\wsl-nanovllm.ps1 -Command "python benchmarks/benchmark_m15_decode.py --model /opt/models/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c --seq-len 32768 --gen-tokens 32 --index-select 1 --cpu-threads 8 --top-k 32 --record-selected-ids --output bench_logs/m21_staticmask_repro_1_0.json > bench_logs/m21_staticmask_repro_1_0.log 2>&1"
.\scripts\wsl-nanovllm.ps1 -Command "python benchmarks/benchmark_m15_decode.py --model /opt/models/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c --seq-len 32768 --gen-tokens 32 --index-select 1 --cpu-threads 8 --top-k 32 --selector-static-mask --record-selected-ids --output bench_logs/m21_staticmask_repro_1_1.json > bench_logs/m21_staticmask_repro_1_1.log 2>&1"
.\scripts\wsl-nanovllm.ps1 -Command "python benchmarks/benchmark_m15_decode.py --model /opt/models/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c --seq-len 32768 --gen-tokens 32 --index-select 1 --cpu-threads 8 --top-k 32 --selector-static-mask --record-selected-ids --output bench_logs/m21_staticmask_repro_2_1.json > bench_logs/m21_staticmask_repro_2_1.log 2>&1"
.\scripts\wsl-nanovllm.ps1 -Command "python benchmarks/benchmark_m15_decode.py --model /opt/models/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c --seq-len 32768 --gen-tokens 32 --index-select 1 --cpu-threads 8 --top-k 32 --record-selected-ids --output bench_logs/m21_staticmask_repro_2_0.json > bench_logs/m21_staticmask_repro_2_0.log 2>&1"
```

For a new static-mask step-4 capture, use another fresh output prefix:

```powershell
.\scripts\wsl-nanovllm.ps1 -Command "nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop -o bench_logs/m21_staticmask_repro_step4_static python benchmarks/benchmark_m15_decode.py --model /opt/models/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c --seq-len 32768 --gen-tokens 8 --index-select 1 --cpu-threads 8 --top-k 32 --selector-static-mask --record-selected-ids --nsys-trace-step 4 --output bench_logs/m21_staticmask_repro_step4_static.json"
.\scripts\wsl-nanovllm.ps1 -Command "nsys stats --force-export=true --report cuda_api_sum,nvtx_sum bench_logs/m21_staticmask_repro_step4_static.nsys-rep"
```

To capture the corresponding serial or pipeline arm, omit `--selector-static-mask`;
add `--kv-pipeline` only for the pipeline arm. Keep the `m21_staticmask_repro_`
prefix for every replay, including the `.nsys-rep` and exported SQLite names.
