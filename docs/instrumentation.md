# Milestone 1: cache instrumentation

Milestone 1 adds observation only. It does not add a CPU cache or alter the
upstream GPU prefix-cache policy.

## Public contract

Pass `enable_cache_metrics=True` to `LLM`. After each public `generate()` call,
`llm.get_cache_metrics()` returns a JSON-serializable dictionary describing that
call. A new `generate()` starts a fresh metrics window; this keeps lookup counts,
prefill/decode timing, and TTFT scoped to the same request batch.

`llm.reset_cache_metrics()` is also available for callers that drive the lower
level `step()` API. With the feature flag at its default `False`, the engine does
not create lookup/model timers, and `get_cache_metrics()` returns `{}`.

## Counter semantics

- `requests`: sequences whose initial prefix lookup was observed.
- `hits` / `misses`: requests with at least one reusable block or none.
- `gpu_hit_blocks`: contiguous complete token blocks found in the upstream GPU
  cache.
- `reused_tokens`: `gpu_hit_blocks * block_size` in this milestone.
- `recomputed_tokens`: prompt tokens not covered by those complete blocks.
- `cpu_hit_blocks`, transfer bytes/times, and evictions remain zero until the CPU
  tier is implemented in later milestones.

A shared tail smaller than one block is never counted as reused. The current
upstream lookup also leaves the final sequence block scheduled even when the
prompt length is block-aligned, as documented in `architecture.md`.

## Timing boundaries

- `lookup_time_ms` uses a CPU wall clock around chained-hash lookup and token-ID
  collision verification.
- `prefill_time_ms` and `decode_time_ms` use CUDA Events around model execution;
  reading the elapsed value synchronizes the end event.
- `ttft_ms` starts immediately before the first scheduler call and ends after
  the first prefill step has sampled and postprocessed the first token.
- `first_token_time_ms` is the same synchronous boundary as `ttft_ms`; it is kept
  explicit so later benchmarks can distinguish it from model-only prefill time.

These values are stage measurements, not a statistical benchmark. Kernel
compilation can dominate the first cold run.

## Reproduce the GPU validation

From the authoritative WSL environment:

```bash
source .venv/bin/activate
export CUDA_HOME=/usr/local/cuda-12.8
export HF_HOME=/opt/models/.cache/huggingface

python benchmarks/validate_instrumentation.py \
  --model /opt/models/Qwen3-0.6B \
  --output benchmarks/results/instrumentation_validation.json

python benchmarks/deterministic_baseline.py \
  --model /opt/models/Qwen3-0.6B \
  --max-tokens 8 \
  --enable-cache-metrics \
  --output benchmarks/results/instrumentation_equivalence.json
```

The first command asserts five real GPU cases before writing output:

| Case | Reused tokens | Recomputed tokens | GPU blocks |
| --- | ---: | ---: | ---: |
| cold miss | 0 | 600 | 0 |
| same prompt | 512 | 88 | 2 |
| one-block aligned prefix | 256 | 344 | 1 |
| 300-token non-aligned prefix | 256 | 344 | 1 |
| completely different prompt | 0 | 600 | 0 |

The cold and same-prompt cases both generated token IDs `[1018, 1019]`. The
separate feature-off and feature-on deterministic baselines both generated
`[151667, 198, 32313, 11, 279, 1196, 1101, 4588]`. Raw measurements are in
`benchmarks/results/`; no values in this document are hand-entered performance
claims.
