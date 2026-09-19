# Project Roadmap

## Objective

Build and evaluate KV-cache-aware scheduling and multi-GPU request placement for nano-vLLM. Every optimization must include correctness tests, a reproducible workload, and before/after measurements.

## Milestone 0: Reproducible Baseline

Deliverables:

- Request-level TTFT, TPOT, and end-to-end latency.
- Step-level prefill/decode throughput.
- JSON and CSV benchmark artifacts.
- Fixed random seeds and documented hardware/software configuration.

Exit criteria:

- Repeated runs produce comparable workload token counts.
- Metrics distinguish prefill from decode.
- Warmup is excluded from the measured run.

## Milestone 1: Latency-Aware Mixed Scheduling

Problem: the upstream scheduler returns a prefill-only batch whenever waiting work is available. Sustained prompt arrivals can delay active decode requests and increase TPOT.

Planned design:

- Reserve a configurable token budget for decode.
- Use the remaining budget for chunked prefill.
- Add limits for maximum prefill chunk size.
- Preserve request fairness with waiting-time accounting.

Evaluation:

- Compare upstream and mixed scheduling on short, long, and mixed prompts.
- Report throughput, P50/P95 TTFT, P50/P95 TPOT, and starvation events.

## Milestone 2: Cost-Aware Preemption

Problem: upstream preemption releases all KV blocks and restarts a sequence from prefill without estimating recomputation cost.

Planned policies:

- LIFO baseline.
- Minimum cached-token victim.
- Maximum reclaimable-block victim.
- Weighted recomputation/fairness cost model.

Evaluation:

- Preemption count.
- Recomputed prompt tokens.
- Reclaimed KV blocks.
- Throughput and tail latency.

## Milestone 3: Prefix Cache Policy

Planned work:

- Prefix cache hit/miss and saved-token telemetry.
- Explicit LRU and LFU eviction policies.
- Cache admission policy for one-off prefixes.
- Reference-count safety tests.

## Milestone 4: Multi-GPU KV-Aware Routing

Build multiple inference workers and compare request placement policies:

- Round robin.
- Shortest queue.
- Most free KV blocks.
- Prefix affinity.
- Combined KV-aware cost model.

Evaluation requires at least two GPUs and reports per-GPU utilization, cache pressure, request imbalance, throughput, and tail latency.

## Stretch Goals

- Cross-GPU KV block migration.
- Tensor-parallel communication profiling and overlap.
- KV cache quantization.
- Fused or vectorized KV cache write kernels.
- Single-GPU MoE inference.
