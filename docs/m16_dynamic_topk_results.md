# M16: decode-only adaptive Top-K pilot

Status: local experiment on `codex/dynamic-topk`. This is not a demonstrated
end-to-end speedup, nor a full RULER result. The adaptive policy remains off by
default.

## Question and implementation

Could a query-dependent historical KV budget reduce CPU gather and H2D work
without lowering retrieval quality? The implementation leaves prefill at 32
historical blocks and changes only decode. Fixed decode caps of 16/24/32 are
available as controls. Adaptive decode first ranks up to 32 blocks with the
existing `r=4` representative selector, then selects the smallest budget from
16, 24, and 32 whose cumulative normalized *router-score* weight reaches
`0.90`. Flat, invalid, or short score lists keep all candidates. Sink 64 and
recent 512 tokens remain outside this historical budget.

This is a heuristic over representative scores, **not** calibrated attention
mass. It does not compare historical blocks with the recent window, so it does
not yet directly implement a reliable “recent-only versus global” decision.

## Performance pilot

Qwen3-4B BF16 (snapshot `1cfa9a7208912126459214e8b04321603b3df60c`),
RTX 3080 Laptop 16 GB, 32,768-token repeated-text prompt, 16 greedy generated
tokens, 36 layers, block 64, `r=4`, sink 64, recent 512, CPU `index_select`
gather, FlashAttention chunk prefill, YaRN 128K, TP=1/eager. Each row launches
a fresh process. Timing is CUDA-synchronized per engine step; the steady median
drops the first four decode steps. The last step's tensor-shape H2D estimate is
not a PCIe hardware counter. `bench_logs/m16_*_32k_run*.json` contains every
step time and each layer's selected K.

| Decode policy | Steady median ms/token by run | Mean K | Selected historical KV H2D MiB/token |
| --- | ---: | ---: | ---: |
| Fixed 32 | 144.41, 148.56 | 32 | 288 |
| Adaptive 0.90 | 147.70, 150.45 | 23.48 | 211.33 average |
| Fixed 24 | 148.95 (one run) | 24 | 216 |
| Fixed 16 | 146.31, 135.09 | 16 | 144 |

Adaptive K made 36 of 540 layer/step selections at K=16, 503 at K=24,
and one at K=32. Its estimated selected-KV transfer was 26.6% below fixed
K=32. All four policies produced the same 16 token IDs on this repeated-text
prompt; that is an engineering check, not a quality result. Run-to-run latency
varied materially, including an 11.22 ms/token spread between the two fixed-16
runs. The adaptive runs were slightly slower than fixed-32; **do not claim a
TPOT speedup** from these data. The performance input is intentionally simple
and does not probe retrieval difficulty.

## Quality pilot and an important correctness fix

Data: [`tturing/ruler-500-llama2`](https://huggingface.co/datasets/tturing/ruler-500-llama2)
revision `09a858d8bb8d38fbe32920857fb884b303fdf978`, a *community*
regeneration with NVIDIA RULER generators and Llama-2 token sizing. We used
the first 10 fixed rows of each file and the unmodified NVIDIA RULER legacy
`string_match_all` metric function from `NVIDIA/RULER` commit
`c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a`. The 8K single-needle prompts
are actually 7,707–7,722 Qwen3 tokens; the nominal 32K single-needle prompts
are 30,723–31,332; the 8K multi-key prompts are 6,690–7,436. Generation is
greedy with a 32-token cap for NIAH and 30 for variable tracking. NVIDIA's
standard NIAH cap is 128, so these are **small matched diagnostics**, not
comparable to published RULER leaderboard numbers or a full `ns eval` run.

| Task | Dense | Fixed K=32 | Adaptive 0.90 |
| --- | ---: | ---: | ---: |
| Single needle, nominal 8K (10) | 100 | 100 | 100 |
| Single needle, nominal 32K (10) | not feasible on 16 GB GPU | 100 | 100 |
| Multi-key, nominal 8K (10) | not run | 100 | 100 |
| Variable tracking, nominal 8K (10) | not run | 94 | 94 |

Scores are percentages from RULER's substring-matching function. They do not
measure exact answer formatting or broad generation quality. The 8K and 32K
single-needle correct seven-digit answer was also the first seven-digit number
in all 10 responses under both sparse policies; the dense 8K control did so
for all 10 as well.
On variable tracking, both sparse policies retrieved the same number of
reference strings for each of the 10 examples (3/5, 4/5, then 5/5 for each
of the remaining eight), not merely the same aggregate 94%.

An initial fixed-K run scored 20/100 on 8K single needle while dense scored
100/100. Investigation found a real M12 lifecycle bug: consecutive
`LLM.generate()` requests in one model instance inherited the prior request's
CPU KV/reps/recent state. The wrong answers repeatedly contained an earlier
sample's number. We added per-new-sequence M12 reset that clears logical
lengths/selection state while retaining allocated buffers. After this fix,
the same fixed-K sample scored 100/100. The pre-fix sparse outputs are invalid
and excluded from the table; this is independent of adaptive K.

## Decision and boundaries

The candidate budget and transfer reduction are real, but end-to-end speedup
is unproven. The small quality sample did not detect a regression, but cannot
establish general quality preservation. Keep adaptive K disabled by default.
The focused M14-layout and M16-budget tests passed (`11 passed`), benchmark
scripts compiled, and `git diff --check` was clean. Real inference runs above
completed without OOM or non-finite-output errors.
The earlier chunked-prefill previous-recent-window omission was fixed in M14:
later chunks now restore the previous recent K/V before attention. The M16
performance benchmark still routes each 4096-token prefill chunk using one
mean-query summary, so reducing prefill Top-K needs separate quality checks.
Next useful experiments would compare adaptive K with fixed K=24 at matched
average transfer, include larger multi-key/tracing/aggregation samples and a
fully pinned official RULERv1 pipeline, then profile why reduced gather/H2D is
not visible in TPOT.

## Reproduction pointers

- Performance: `benchmarks/benchmark_m15_decode.py --seq-len 32768 --gen-tokens 16 --index-select 1 --top-k 32 --output <path>`; add `--dynamic-top-k --dynamic-mass 0.90` for adaptive, or change `--top-k` to 24/16 for fixed controls.
- Quality inference: `benchmarks/benchmark_m16_ruler.py --input <downloaded-validation.jsonl> --output <pred.jsonl> --top-k 32 --max-tokens 32 --limit 10`; add `--dynamic-top-k` or `--dense` as needed. Use `--max-tokens 30` for variable tracking.
- Quality scoring: `benchmarks/score_m16_ruler_subset.py --predictions <pred.jsonl> --constants <pinned-RULER>/scripts/eval/synthetic/constants.py --task niah` (or `variable_tracking`).
- Local raw logs/predictions/data are ignored under `bench_logs/`; the source, scorer, adapter, and revision identifiers above are the reproducible record.
