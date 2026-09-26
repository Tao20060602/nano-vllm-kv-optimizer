# M17: prefill budget and CPU-path experiments

Status: local experiment on `codex/m17-prefill-topk`. The prefill-only budget
override is opt-in; the default remains the previous Top-K behavior. No new
end-to-end prefill speedup is promoted as a default configuration.

## Controlled setup and timing boundary

Qwen3-4B snapshot `1cfa9a7208912126459214e8b04321603b3df60c`, BF16,
YaRN, RTX 3080 Laptop 16 GB, TP=1, 32,768-token fixed repeated-text prompt,
4096-token chunks, 64-token retrieval blocks, `r=4`, sink 64, recent 512,
one mean-Q summary per chunk, FlashAttention-2, decode cap fixed at 32. A
separate `sparse_prefill_top_k` can lower prefill K while `sparse_top_k=32`
continues to size buffers and `sparse_decode_top_k=32` remains unchanged.

Each fresh-process run synchronizes before and after every engine step and
generates one token. `prefill_wall_ms` is the sum of positive-schedule steps;
the last such step also samples the first output token. It excludes model
initialization and tokenization, and is neither pure model-only prefill nor
full user-facing TTFT. We emphasize the sum of the seven *later* chunks because
the first chunk does not retrieve history and varied substantially by run.

| Prefill K | Later seven chunks, run 1 / run 2 (ms) | Mean (ms) | Change vs K=32 | Selected-history H2D per later chunk* |
| --- | ---: | ---: | ---: | ---: |
| 32 | 13328.77 / 12275.93 | 12802.35 | baseline | 288 MiB |
| 24 | 12017.90 / 11140.70 | 11579.30 | -9.55% | 216 MiB |
| 16 | 11848.52 / 11103.90 | 11476.21 | -10.36% | 144 MiB |

*Shape-derived payload for 36 layers, not a PCIe counter. K=24 and K=16
were both faster than K=32 in these two runs, but their 0.9% separation is
too small to establish that K=16 is faster than K=24. Full positive-step wall
means were 16.65 / 15.13 / 14.34 s for K=32/24/16, respectively; the
unaffected first chunk contributed 2.71-4.96 s across individual runs, so
the full-wall percentages are less interpretable.

## Matched quality screen: why lower K is not the default

The first five 32K single-needle examples scored 5/5 at all three budgets.
The first three 32K multi-key examples scored 1/3 at K=32 and 0/3 at both
K=24 and K=16. In example `44391`, K=32 produced the correct `1908841`,
K=24 produced `1908871`, and K=16 produced `1908`. The other two multi-key
examples were already wrong at K=32. This small screen cannot estimate general
quality or isolate model versus sparse-attention limitations, but it is a
concrete regression against the matched K=32 control. **Do not enable lower
prefill K by default or claim quality preservation.**

The data are the same community-generated legacy RULER files and pinned
NVIDIA metric function documented in `docs/m16_dynamic_topk_results.md`.
Generation used greedy decoding with a 32-token cap and fixed decode K=32.
This is a small matched diagnostic, not an official full RULER score. A 32K
dense Qwen3-4B control was not run on the 16 GB GPU.
The first K=24 quality run's terminal summary mistakenly printed prefill K=32
because that print field was still hard-coded; its model configuration already
passed the requested K=24. The summary field was corrected before K=16 ran.

## CPU gather and profiler findings

At K=32, copying selected CPU blocks directly into pinned staging with
`torch.index_select(..., out=...)` preserved selected block IDs and toy
attention output exactly. Across two 32K runs, instrumented CPU gather over
all 36 layers and seven later chunks was 341.49/403.59 ms with the old path
versus 231.67/255.38 ms with index-select (mean -34.6%). But later-chunk
wall times were 11324.68/11975.32 ms versus 11302.42/11796.60 ms (mean
-0.9%, within run variation). The gather improvement is real locally; an
end-to-end prefill speedup is **not established**. Keep the existing opt-in
`sparse_gather_index_select` flag, now supported in prefill as well as decode.

A PyTorch-profiler probe of one later 4096-token chunk in a 16K request
reported approximately 722 ms in GEMM kernels, 305 ms in the FA2 forward
kernel, 255 ms in pageable D2H copies, and 25 ms in pinned H2D copies. These
are exclusive device-activity categories from a *profiled* step, not shares
of uninstrumented prefill wall time; profiler overhead made that engine step
3894 ms. The next plausible prefill systems experiment is overlap or staging
of full-KV D2H, not further shaving the small selected-block CPU gather.
`nsys` and `ncu` were not installed in this WSL environment.

Before trying another Q summary or a smaller budget, the next narrow
diagnostic should locate the target block in failing multi-key example
`44391` and record its rank/selection per prefill chunk and layer. That can
separate a routing miss from a downstream attention/model error. If routing
is the cause, evaluate a Q-summary change at the *same* K first; the single
M14 four-summary needle result did not establish an improvement.

A trial direct copy from GPU blocks into the existing pageable CPU KV buffer
passed toy exactness checks but yielded 11386.84/11748.77 ms for later chunks,
overlapping the legacy 11324.68/11975.32 ms. It was reverted rather than
retained as another unproven switch.

## Validation and provenance

Focused M14/M16/M17 tests: 19 passed after reverting direct D2H. One attempted
K=24 repeat crashed in a WSL/PyTorch NCCL background thread before writing a
result; the successful retry is run 2 above. A Qwen3-0.6B functional run was
excluded from all Qwen3-4B comparisons. Benchmark JSON files record the
pre-existing `593d741` HEAD because experiments ran on an uncommitted M17
working tree; the tracked M17 source and this report are the reproducible
implementation record.
The first K=32 log predates the explicit `rope_mode` JSON field. The script
at recorded HEAD `593d741` hard-coded `rope_scaling_override=YARN`; all
subsequent 4B logs explicitly record `rope_mode=yarn`.

Raw logs and predictions are ignored under `bench_logs/`:

- `m17_prefill_k32_32k_run1.json` and `m17_prefill_4b_k32_32k_run2.json`
- `m17_prefill_4b_k24_32k_run{1,2}.json`
- `m17_prefill_4b_k16_32k_run{1,2}.json`
- `m17_quality_single32k_prefill{32,24,16}.jsonl`
- `m17_quality_multikey32k_prefill{32,24,16}.jsonl`
- `m17_gather_{legacy,index}_32k_run{1,2}.json`
- `m17_direct_d2h_32k_run{1,2}.json` (reverted trial)
- `m17_prefill_profile_4b_16k_step1.json`

Reproduction example (run one K per fresh process, changing only
`--prefill-top-k`):

```bash
cd /opt/nano-vllm
source .venv/bin/activate
export HF_HOME=/opt/models/.cache/huggingface
python benchmarks/benchmark_m14_prefill.py --seq-len 32768 \
  --chunk-size 4096 --query-segments 1 --backend flash \
  --prefill-top-k 32 --output bench_logs/m17_repeat_k32.json
```
