# Line A: decode-routing diagnosis and relative-threshold selection

Status: in progress (2026-10-10). Opt-in code is implemented and default-off;
the A/B (`bench_logs/lineA/ab_relative.json`, gitignored) is running.
This document is the handoff for any agent continuing the work.

## Question

M22 left a quality gap: 8K `niah_multikey_2` dense 100 vs sparse 80, and 32K
distractors 0/10 with no dense control. Is the miss in prefill routing,
decode routing, or the model itself?

## Reproduction inputs (CPU-only, no GPU needed)

RULER is pinned at `c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a`, seed 42,
Qwen3-4B HF tokenizer, `max_seq_length` 8192, 20 samples:

```bash
cd /tmp/opencode/RULER/scripts/data  # or any RULER checkout at c3f5e3b
python prepare.py --save_dir <out> --task niah_multikey_2 \
  --tokenizer_path /home/tmz/models/Qwen3-4B --tokenizer_type hf \
  --max_seq_length 8192 --model_template_type base \
  --num_samples 20 --random_seed 42
```

Caveat: regenerated content is length-matched to M22 (7994-8051 prompt
tokens) but not bit-identical (wonderwords version shifts the RNG chain).
It is a same-distribution diagnostic set, not a paired M22 rerun.

## Findings so far (Qwen3-4B BF16, YaRN x4, chunk 4096, Top-32, r=4, sink 64, recent 512)

1. Baseline on the regenerated 20 rows reproduces the failure mode: 12/20,
   wrong on 6258/8786/10273/18422/9476/9546/7980/5200 (deterministic rerun
   confirmed identical rows).
2. Splitting targets by chunk: in-chunk targets (block >= 64, seen by causal
   within-chunk attention, no selection needed) score 6/7; history-zone
   targets score 6/13. Prefill Top-32 membership does not separate right
   from wrong (correct {18,28,22,24,25,29} vs wrong {16,17,22,17,20,21,22}
   selecting layers).
3. Decode-phase target hit rate separates cleanly: wrong rows 0.22-0.36,
   right rows 0.39-0.62 mean across layers/steps. Extreme case 7980:
   prefill selects the target in 21/36 layers, decode in ~0.
4. Decode score margins (post-hoc, `diagnose_target_margin`): wrong rows sit
   4.5-14.8 raw points below the Top-32 cutoff (mean rank 50-65 of ~125);
   right rows sit at or above it (gap -11.6 to +2.5). Mechanism: prefill
   routes with a 4096-token mean Q (`prefill_query_summaries`), decode
   routes with a single-token Q (`q[0]`); the two disagree.

## Implemented (all default-off, run with `a=None` to reproduce baseline)

- `M12Config.decode_relative_a` / `decode_relative_max_blocks` (=48),
  plumbed as `Config.sparse_decode_relative_a` /
  `Config.sparse_decode_relative_max_blocks` through `ModelRunner`.
  Decode passes `use_relative=a is not None` to `_gpu_select`.
- Relative rule: keep blocks with raw score `>= best + ln(a)/scale`,
  i.e. attention weight `>= a * max`; capped at max blocks; degenerate
  all-masked fallback to fixed top-k; raises on the graph path.
  With scale `1/sqrt(128)`: `a=0.1` ~= beta 26, `a=0.01` ~= beta 52.
- Diagnosis hooks: `record_q` / `q_history` and `diagnose_target_margin`
  (mirrors the flat scoring branch; static-mask/graph branches excluded).

## Running the A/B

```bash
cd /opt/nano-vllm  # or this worktree; one LLM per process (NCCL)
.venv/bin/python bench_logs/lineA/ab_relative.py \
  --arms none,0.01 --output bench_logs/lineA/ab_relative.json
```

Each arm runs in its own subprocess. Metrics per row: reference substring
in 32-token greedy output, mean last-step selected-history tokens.
Next arms if `a=0.01` saturates the cap: `0.1`, `0.001`.

## Open items

- A/B result pending at time of writing; decide from it whether the
  relative rule, Q-window averaging, or a per-layer policy comes next.
- `bench_logs/` is gitignored: scripts and JSON there are NOT in git;
  methodology changes must be mirrored here.
- Do not claim quality preservation: the set is 20 rows, one seed, one
  prompt distribution; 32K dense control still missing.
