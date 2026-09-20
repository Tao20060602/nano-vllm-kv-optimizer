# Milestone 6: benchmark methodology

This document describes how the Milestone 6 CPU-backed prefix-reuse benchmark was
run, how each number in `benchmarks/results/m6/report/` is defined, and which
scenarios were intentionally not measured.

## Hardware and software environment

- Host: Windows desktop with WSL2 (`NanoVLLM-Ubuntu` distribution).
- Kernel: `Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.39`.
- GPU: NVIDIA GeForce RTX 3080 Laptop GPU, 16 GiB.
- CUDA driver visible inside WSL: CUDA UMD 13.4 (Windows host), build tree
  compiled against CUDA 12.8 (`CUDA_HOME=/usr/local/cuda-12.8`).
- PyTorch: `2.7.1+cu128`.
- Python: CPython 3.12 inside `/opt/nano-vllm/.venv`.
- GPU was idle before every mode (`nvidia-smi` showed 0 MiB used, 0% util);
  only one benchmark mode was run at a time, never in parallel.

## Model

- Path: `/opt/models/Qwen3-0.6B` (Qwen3 0.6B, HF format, bf16 as shipped).
- `max_model_len=4352` so the largest measured prompt (4112 tokens + 1 output)
  stays under the limit.
- `enforce_eager=True`, `tensor_parallel_size=1`, one scheduled sequence at a
  time (same constraints as Milestone 5).
- KV block size is 256 tokens. One KV block for this model/dtype is
  29,360,128 bytes (28.0 MiB).

## Test input construction

All prompts are deterministic integer token IDs (no tokenizer), built by
`benchmarks/benchmark_prefix_reuse.py`:

- `exact_prompt(prefix, suffix, namespace)`: a prefix of `prefix` tokens from
  one token-id namespace followed by `suffix` tokens from a different namespace.
  Across modes the exact same token IDs are generated for a given
  `(prefix, suffix, namespace)`, so cold / gpu / cpu_pageable / cpu_pinned all
  see the identical input.
- `partial_prompt(total, shared, namespace, variant)`: `shared` tokens from a
  shared namespace, then `total-shared` tokens that vary with `variant`. The
  shared prefix is token-identical across variants; the tail differs.

Prefix lengths are 256, 512, 1024, 2048, 4096 (all multiples of the 256-token
block size, so the whole prefix is block-aligned). Two request shapes:

- Break-even point: `suffix=16`, `max_tokens=1` (measure TTFT only).
- Full request: `prefix=1024, suffix=64, max_tokens=32` (measure TTFT and wall
  time including 32 decode steps).

Partial-reuse experiment: total prompt 2064 tokens (8 full blocks + 16 tail),
with shared prefix of 2, 4, 6, 8 blocks -> reuse ratios 25%, 50%, 75%, 100%.

## Warmup and repetition

- 5 warmup iterations + 20 measured iterations per point (hard floor enforced
  by the script: it exits if `--warmups < 5` or `--repetitions < 20`).
- Warmup iterations are kept in the raw JSON with `"warmup": true` but are
  excluded from `generate_report.py` statistics.
- For non-cold modes the prompt is sent once as a "prime" before the warmup
  loop to populate the cache. For CPU modes the GPU prefix cache is then
  explicitly cleared so the prime's KV lands only in the CPU store.

## Cache isolation between modes and iterations

- `cold`: GPU prefix cache is cleared before every measurement. CPU cache is
  disabled entirely.
- `gpu`: reusable GPU cache enabled; GPU cache is NOT cleared between
  iterations, so every measured iteration is a true GPU hit. The prime
  populates it.
- `cpu_pageable` / `cpu_pinned`: CPU store enabled. GPU prefix cache is
  cleared before every measured iteration (after the prime), so each
  measurement exercises the CPU -> GPU restore path rather than a GPU hit.
  The CPU store itself is intentionally NOT cleared between iterations; that
  is what makes the CPU hit possible.
- `cpu_partial_pinned`: same CPU pinned store; GPU cache cleared after every
  iteration. Each variant changes only the non-shared tail, so the shared
  blocks remain in the CPU store.

## CUDA synchronization and timing

Every measured call is wrapped in:

```
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
start = time.perf_counter()
output = llm.generate([prompt], sampling, use_tqdm=False)
torch.cuda.synchronize()
wall_ms = (time.perf_counter() - start) * 1000
```

The pre/post synchronize ensures that async CUDA work (prefill kernels,
H2D copies, decode) has actually completed before the wall clock is read.
Per-phase timings come from `llm.get_cache_metrics()` (CPU lookup, H2D load,
prefill) and `llm.get_cpu_cache_stats()` (resident bytes, throughput).

## Timing boundaries and TTFT definition

`ttft_ms` (time to first token) is measured inside the engine from request
arrival to the first sampled token logits, and includes:

- CPU prefix lookup (`lookup_time_ms`): hash lookup in the ContextDB index;
- H2D restore (`load_time_ms`): synchronous CPU -> GPU copies of matched full
  blocks (0 for cold and gpu modes);
- actual prefill (`prefill_time_ms`): prefill forward on the tokens that were
  not reused;
- plus engine scheduling and the first decode step to produce logits.

`wall_time_ms` is the end-to-end `generate()` wall clock (TTFT + all decode
steps), used only for the `max_tokens=32` point. `h2d_bytes` is the cumulative
bytes transferred CPU -> GPU during restore for that request.

## pageable vs pinned configuration

- `cpu_pageable`: `cpu_cache_pinned=False`, KV blocks live in pageable host
  memory; H2D copies go through staging.
- `cpu_pinned` / `cpu_partial_pinned`: `cpu_cache_pinned=True`, blocks are
  allocated with `torch.cuda.host_alloc`-style pinned memory for lower
  transfer latency.
- CPU store capacity: 64 blocks (`cpu_cache_capacity_bytes = 64 * bytes_per_block`
  = 1,759,607,552 bytes ~ 1.64 GiB), i.e. up to 64 x 256 = 16,384 cached
  tokens. LRU eviction applies; no capacity override (`--cpu-capacity-blocks`)
  was needed because pinned memory did not cause WSL memory pressure.

## Correctness guards in the benchmark itself

Every measured record asserts (and would crash, not silently miscount):

- `metrics.reused_tokens` equals the expected reused token count;
- `metrics.prefill_executed_tokens` equals `prompt_tokens - expected_reused`;
- for the partial mode, `metrics.cpu_hit_blocks` equals the expected shared
  block count.

Generated `output_token_ids` are stored for every record so post-hoc token
comparison is possible.

## Why 8192 prefix is not run

`max_model_len=4352`. An 8192-token prefix plus suffix would exceed the model
context window for this configuration, and doubling the GPU KV cache budget
for one larger data point would change the memory layout compared with the
other four modes. The raw JSON explicitly records
`"not_run_prefix_tokens": [8192]` rather than fabricating a value.

## Artifacts and reproducibility

- Raw per-iteration JSON (one per mode):
  `benchmarks/results/m6/{cold,gpu,cpu_pageable,cpu_pinned,cpu_partial_pinned}.json`
- Generated report:
  `benchmarks/results/m6/report/{summary.json, summary.csv, conclusions.json, ttft_break_even.svg}`
- Benchmark driver: `benchmarks/benchmark_prefix_reuse.py`
- Report generator: `benchmarks/generate_report.py`

To reproduce from scratch:

```bash
python benchmarks/benchmark_prefix_reuse.py --model /opt/models/Qwen3-0.6B \
    --mode cold --output benchmarks/results/m6/cold.json
# repeat for gpu / cpu_pageable / cpu_pinned / cpu_partial_pinned
python benchmarks/generate_report.py benchmarks/results/m6/*.json \
    --output-dir benchmarks/results/m6/report
```

All numbers in the report are derived from the raw JSON by
`generate_report.py`; no summary figure was hand-edited.