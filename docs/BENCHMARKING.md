# Benchmarking Methodology

## Metrics

- **TTFT**: request registration to first generated token.
- **TPOT**: time from the first token to completion divided by the number of remaining generated tokens.
- **E2E**: request registration to completion.
- **Prefill throughput**: scheduled prefill tokens divided by total prefill step time.
- **Decode throughput**: scheduled decode tokens divided by total decode step time.

TTFT in the initial offline benchmark includes queueing because all requests are registered before execution. Later online-arrival benchmarks will record arrival schedules explicitly.

## Required Run Metadata

Every reported experiment should include:

- Git commit.
- GPU model and count.
- Driver and CUDA versions.
- PyTorch, Triton, Transformers, and FlashAttention versions.
- Model and dtype.
- Tensor-parallel size.
- KV cache block size and GPU memory utilization.
- Request count and input/output length distribution.
- Warmup configuration and random seed.

## Workload Families

1. Short prompts and short outputs.
2. Long prompts and short outputs.
3. Short prompts and long outputs.
4. Mixed prompt and output lengths.
5. Shared-prefix workloads at multiple repetition ratios.
6. KV pressure workloads that trigger preemption.

## Comparison Rules

- Use identical prompt token IDs and requested output lengths.
- Disable EOS or persist sampled outputs when comparing scheduling policies.
- Run warmup separately from measurement.
- Report at least three measured repetitions for final results.
- Include both central tendency and P95 latency.
- Treat GPU temperature, clocks, and competing processes as experimental conditions.

## Profiling Workflow

1. Use application metrics to identify the problematic workload.
2. Use PyTorch Profiler or Nsight Systems to separate CPU, launch, compute, memory, and NCCL time.
3. Use Nsight Compute only for selected kernels.
4. Re-run the same workload after optimization.
5. Report regressions as well as improvements.
