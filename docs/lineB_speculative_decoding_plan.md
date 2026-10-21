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

## Findings that changed the design (2026-10-13)

The acceptance probe passed: Qwen3-0.6B teacher-forced agreement with the
4B target is **70.8%** on 8K RULER prompts -> ~2.1 accepted tokens per pass
at K=4, ~2.8 at K=8.

But a second measurement killed the **lossless** variant. Consecutive
decode tokens only reuse **32.6%** of their selected blocks (matches M13),
so a lossless per-query verification would need the **union** of the K
selections for its transfer:

| K | union blocks | vs single (32) |
|---|---:|---:|
| 1 | 32.0 | 1.00x |
| 2 | 53.6 | 1.68x |
| 4 | 85.3 | 2.67x |
| 8 | 127.2 | 3.98x |

At K=4 the union is 85 blocks for ~2.1 tokens = 40 blocks/token vs 32 for a
single step: **the transfer would grow ~1.3x, not shrink.** The amortization
premise fails because the top-32 set reshuffles each step (the attention
*output* is robust, the selected *set* is not).

**Decision: implement the approximate variant** - all K verification queries
share ONE selection (max-reduce over their scores), so transfer stays ~32
blocks per pass -> ~15 blocks/token at K=4. This is no longer bit-identical
to non-speculative sparse decode and must pass the same kind of quality
screen int8 did.

## Implementation progress

- Step 1 DONE: `M12LayerRuntime.verify_tokens` (shared selection, packs
  `[selected_hist | sink | previous recent | chunk(Tq, causal)]`, one exact
  attention) and `commit_verified(n)` / `rollback_verified()`.
  Correctness test `bench_logs/lineB/test_verify.py`: Tq=1 verify equals the
  single-token `decode` (block ids and valid_len identical; output differs by
  0.003 fp noise because decode uses the batched path and verify the per-head
  causal path). Tq=3 + commit(2) behaves.
- Step 2 TODO: draft runner (Qwen3-0.6B dense, bounded KV window, cache
  rollback on rejection).
- Step 3 TODO: speculative loop (greedy accept prefix + bonus; commit/rollback).
- Step 4 TODO: quality screen (290-row RULER set) + TPOT/acceptance sweep.

## First end-to-end result (2026-10-20, 8K niah_single row, greedy, K=4, window=8192)

Same-process timing (prefill excluded), base vs spec:

| arm | ms/token | rounds | tokens/pass | accept/proposal |
|---|---:|---:|---:|---:|
| baseline sparse decode | 126.2 | - | 1 | - |
| speculative (K=4) | **124.8** | 8 | 2.88 | 0.47 |

Breakdown per round: draft **175 ms**, verify **180 ms** (5 query tokens).
The verification side works exactly as theorized: 5 tokens for 180 ms is
**36 ms/token** vs 126 ms for a normal step, because one verify pass
amortizes both the history H2D **and** the model-weight read. If the draft
were free this would be ~1.6x. **The draft cost (175 ms/round, 35 ms/step)
is the entire loss.**

The hand-written transformer draft measured 18.9 ms/step standalone with
SDPA at 8K, so there is a ~1.8x gap inside the loop that is not yet
explained (likely contention/thermal/overhead). Closing it to ~10 ms/step
would make the total ~ (50 + 180)/2.88 = 80 ms/token, ~1.6x.

Generated token IDs were **identical** to the baseline on this row, so the
shared-selection approximation did not change the answer here (1 row only;
the 290-row quality screen is still required).

## Lossless verification (2026-10-20)

Because the shared selection is approximate (quality screen below), a
lossless variant was added:

- `_gpu_select_per_query`: per-query top-k over the same frozen history;
  the union is gathered once (`verify_union_blocks` sizes the buffers).
- `_fused_attention(attn_mask=...)`: per-query mask over the union so each
  query attends exactly to its own selected blocks + sink/recent + causal
  chunk. This reproduces non-speculative sparse decode exactly.

Correctness (synthetic runtime, `test_verify.py`):
- union equals the union of three sequential single-token selections
  (51 blocks); per-row output diff <= 0.015 (fp reduction order only).
- End-to-end 8K generation: token IDs **identical** to baseline.

Performance (8K, K=4, window 8192):

| arm | ms/token | note |
|---|---:|---|
| baseline | 131.3 | |
| lossless spec | 138.8 (0.95x) | verify 201 ms/round, draft 193 ms/round |

The lossless verify costs more than the shared one (union transfer), and the
draft still dominates. Break-even needs a draft round <= ~80 ms (currently
193 ms). With a 50 ms draft round this would be ~1.5x.

## Quality screen: approximate (shared) mode (2026-10-20, 290 rows)

| task | base % | spec % | delta | token-eq |
|---|---:|---:|---:|---:|
| 8k multikey (100) | 77.0 | 76.0 | -1.0 | 22 |
| 8k single (60) | 100.0 | 100.0 | 0.0 | 35 |
| 8k vt (60) | 85.7 | 83.7 | -2.0 | 19 |
| 16k multikey (40) | 27.5 | 27.5 | 0.0 | 8 |
| 16k single (30) | 100.0 | 100.0 | 0.0 | 14 |
| total (290) | 79.1 | 78.4 | **-0.75** | 98 (33.8%) |

Score changed on 20 rows: 7 improved, 13 worsened. The shared selection is
**not quality-neutral** (unlike int8). Hence the lossless variant above.

## Round profile (2026-10-20, 8K, K=4, shared mode)

Stable round (~280 ms total, ~2 tokens):

| segment | ms | note |
|---|---:|---|
| draft (5 steps) | 129 | 25-26 ms/step |
| verify selector | 16-18 | |
| verify gather | 28-30 | CPU |
| verify pack (H2D + dequant) | 56-63 | transfer |
| verify attention | 7 | |
| verify model rest (qkv/MLP/lm_head) | 40-47 | |

**No async opportunity exists.** The per-layer chain is
`qkv -> rope -> selector -> gather -> H2D -> attention -> MLP`, and the next
layer's selector needs the previous layer's full output; draft and verify are
strictly ordered too. Every stage depends on the previous one.

The draft's 25 ms/step is **launch-overhead bound**: the 0.6B weights are
~1.2 GB (≈3 ms at HBM speed), 8K attention ≈1 ms, lm_head ≈0.6 ms, so the
theoretical floor is ~5 ms/step. 28 layers x ~5 kernels ≈ 150 launches/step
at WSL launch cost explains the gap. CUDA-graph capture of one draft step is
the lever: ~5-8 ms/step would put the draft at 30-40 ms/round and the total
at ~1.3x (lossless ~1.15-1.2x).

## Static-graph draft engine (2026-10-20)

`nanovllm/sparse/static_draft.py` implements a minimal Qwen3-0.6B forward
with a **preallocated static KV cache**, captured once with
`torch.cuda.CUDAGraph`:

- cache `[layers, window, kv_heads, head_dim]`, written at index `pos`;
- attention over the full window with a mask `arange(W) > pos` (pos is a
  static tensor, so the shape/addresses never change);
- static input/output buffers; one graph replay per token;
- prefill runs once through HF (`logits_to_keep=1` to avoid materializing
  8K x 151936 logits) and copies its K/V into the static cache.

Verified against HF greedy: 10/10 tokens identical on a 401-token prompt.
Step time: **9.8 ms** including the argmax sync (vs ~25 ms for the
transformers draft), i.e. ~2.5x faster.

Note: keeping the HF model resident for repeated prefills costs 1.2 GB and
was measured to slow the target verify (293 vs ~200 ms/round); dropping it
after prefill restores the target speed.

## Final end-to-end results (8K, K=4, window 8192, greedy)

Paired fresh-process measurement (base, spec, spec, base):

| arm | ms/token | speedup | tokens vs baseline |
|---|---:|---:|---|
| baseline sparse | 131.6 / 126.9 (mean 129.2) | 1.00x | - |
| **speculative, lossless** | **84.3 / 83.5 (mean 83.9)** | **1.53x / 1.55x** | **identical** |
| speculative, shared (approx.) | 90.5 / 93.1 | 1.41x / 1.37x | identical on this row |

Round breakdown (lossless): draft 39 ms, verify 178-202 ms, 2.88 tokens/pass,
47% acceptance. The lossless variant is both quality-safe and faster than the
shared one in these runs.

The 290-row quality screen for the shared mode showed -0.75 points; the
lossless mode is bit-identical by construction, so it supersedes it.

## Long context: ring-buffer window (2026-10-21)

The static cache is now a ring buffer over absolute positions
(`slot_pos`, write at `pos % W`), with the mask
`valid = (slot_pos >= 0) & (slot_pos <= pos) & (slot_pos > pos - W)`.
Prefill is self-contained (chunked, chunk-causal, no HF model), so the draft
no longer needs a second model in memory.

16K single-needle row, lossless, K=4:

| draft window | accept | tok/pass | ms/token | speedup | output |
|---|---:|---:|---:|---:|---|
| 8192 | 0.25 | 2.00 | 112.9 | 1.09x | identical |
| 17408 | 0.47 | 2.89 | 81.5 | **1.52x** | identical (prefix) |

The window is a memory/acceptance trade-off: a window smaller than the
context keeps draft memory constant but loses acceptance; matching the
context restores it at ~1.95 GB of cache. Baseline was 123.5 ms/token.

## K sweep at 8K (lossless, static draft, 24-32 tokens)

| K | ms/token | speedup vs 127.5 | tokens/pass |
|---:|---:|---:|---:|
| 2 | 83.4 | 1.53x | 2.18 |
| 3 | 71.1 | 1.79x | 2.89 |
| 4 | 74.0 | 1.72x | 2.88 |
| 5 | **64.6** | **1.97x** | **3.43** |

Larger K keeps helping at ~50% acceptance because the verify pass amortizes
both the H2D and the weight read; the draft step is only ~10 ms. K=6/8 could
not be measured: the union buffer grows to 2.1-2.7 GB (union blocks =
(K+1)*32) and the GPU entered a fault state during those attempts.

This suggests the default K should be larger than 4, and that adaptive K
should raise K when acceptance is high and lower it when the draft window is
smaller than the context (16K with an 8K window gave accept 0.25 and only
1.09x).

## Risks / open points

- **Draft speed is the whole game.** Until the draft step drops well below
  the per-round verify saving, speculative decoding is break-even. CUDA-graph
  capture of the draft step (static padded cache) is the obvious next move.
- **Quality**: the shared selection is approximate; run the 290-row RULER
  screen before claiming anything. One row matched exactly.
- **Window size matters**: at 4K window the 8K RULER draft acceptance was
  3% (needle outside the window); at 8K it was 47%. Window >= task context
  is needed for high acceptance, which raises draft KV to ~0.9 GB at 8K.
- **Rollback**: implemented via deferred commit (`commit_verified`); the
  target's CPU history only stores prefill tokens, so rejection only rolls
  back the recent window and `valid_len`.
- **Memory**: target sparse runtime ~9 GB + draft 1.2 GB + draft window KV
  (0.9 GB at 8K) + activations; fits 16 GB at 8K but not at 64K+ with a
  full-context window (hence the bounded window).
- **Timing noise**: WSL; use paired fresh-process runs and several prompts.
- **Scope**: greedy only; no temperature/rejection sampling in v1.
