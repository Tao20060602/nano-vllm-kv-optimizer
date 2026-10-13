# Line B: speculative decoding for NanoKV (Plan A)

Status: planning + first validation step (2026-10-13). Nothing implemented yet.

## Why

Line A profiling established the decode bottleneck is the per-token CPU->GPU
H2D transfer of selected history (~35 ms of ~85 ms/token with int8), not the
attention compute (~7 ms). Speculative decoding amortizes exactly that: the
same history is transferred once for K verified tokens instead of K times.
With a greedy target the output is bit-identical to non-speculative decoding
(lossless), unlike int8 quantization.

## Design

- **Target** Qwen3-4B with the existing sparse CPU-offload runtime, extended
  to **multi-query decode** (verify K draft tokens in one pass).
- **Draft** Qwen3-0.6B, dense, **bounded KV window** (sink + last N tokens).
  A full-context draft KV would reintroduce the memory problem: the 0.6B KV
  is only ~22% smaller than the 4B's (~114 vs ~147 KB/token), so it cannot be
  kept in full. A bounded window keeps draft memory constant (~456 MB at
  N=4096) regardless of context length; the target still sees full context,
  so correctness is unaffected and only the acceptance rate degrades.
- Greedy verification (accept longest matching prefix; sample one bonus token
  from the target when all K are accepted).

## Acceptance-rate validation FIRST (no engine changes)

Before any implementation, measure how often the 0.6B greedy continuation
matches the 4B's on the project's own prompts. If the token-level agreement
is low, the whole idea is dead and no code should be written.

- Script: `bench_logs/lineB/acceptance_probe.py` (to be written).
- Metric: for each prompt, greedy-generate G tokens with both models; report
  agreement per position and the mean accepted-prefix length for K in {1,4}.
- Go/no-go: proceed if the expected accepted tokens per verify pass is
  meaningfully > 1 (e.g. draft agreement > ~50% for the first few tokens).

## Implementation steps (only if the probe passes)

1. **Multi-query sparse decode** in `M12LayerRuntime` (`decode_multi`).
   Generalize the single-query decode to Tq tokens: union of selected blocks
   over the Tq queries, one gather + H2D, one causal attention over
   `[selected_hist | sink | recent | chunk(causal)]` — structurally the same
   packing `prefill_chunk` already uses. Correctness oracle: K sequential
   single-token decodes must produce the same tokens as one Tq=K pass.
2. **Draft runner**: Qwen3-0.6B dense forward with a windowed KV (sink +
   recent N), correct global positions for RoPE. Verify against a dense
   reference on short prompts.
3. **Speculative loop** (greedy): draft K, verify K+1 logits, accept prefix,
   roll back target history and recent-window state for rejected tokens.
   Acceptance/rollback must leave the target state identical to the
   non-speculative path at every step.
4. Sweep K and N; measure acceptance, TokensPerPass, wall ms/token, and
   bit-identical output vs baseline.

## Risks / open points

- **Rollback**: `decode_multi` appends K tokens to the target CPU history and
  updates sink/recent; rejected tokens must be rolled back exactly. Cleanest
  is to commit history/recent only after acceptance, or snapshot and restore.
- **Memory**: target 8 GB + draft 1.2 GB + draft window KV 0.46 GB + target
  runtime 0.6 GB + activations ~2 GB ~= 12.3 GB, constant in context length
  -> fits the 16 GB 3080 if measured peaks agree.
- **Draft prefill**: the draft must prefill the prompt (0.6B over up to 128K
  tokens) with the windowed KV; cost is a one-time prefill.
- **Timing noise**: WSL; report paired fresh-process comparisons.
- **Scope**: greedy only; no temperature/rejection sampling in v1.
