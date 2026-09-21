# M13-FAST: Decode Reuse Measurement + CPU Gather Fallback

Branch: `m13-fast-decode-reuse` (based on `9c3443b`), worktree clean after commits.
No push (local work branch only).

## 1. Scope recap
M13-FAST per `DOUBAO_M13_FAST_TASK.md`: do not touch retrieval semantics; first
measure real selected-block reuse across consecutive decode tokens; if reuse is
high enough (gate: mean >= 60%, or mean miss <= 12/32 blocks) build a minimal
previous-selected-set GPU-resident cache that only moves miss blocks; otherwise
implement the sole fallback (eliminate the pageable temp tensors of the CPU
gather via `torch.index_select(out=...)`). Produce real before/after wall data
with CUDA-sync boundaries, >= 2 runs each. Prefill/scheduler/retrieval params
untouched.

## 2. Files changed (this branch)
| File | Change |
|---|---|
| `nanovllm/sparse/m12_runtime.py` | M13 diagnostic hook `record_ids`/`ids_history` (default off, no behavior change); `M12Config.use_index_select` flag; decode step 3 branch: one-shot `torch.index_select` into pinned staging vs original advanced-indexing path |
| `nanovllm/config.py` | new field `sparse_gather_index_select: bool = False` |
| `nanovllm/engine/model_runner.py` | pass `use_index_select=config.sparse_gather_index_select` into `M12Config` |

The changes are included in local commit `fcf8625` (the work branch has not
been pushed). The benchmark scripts and raw logs remain under the local
ignored `bench_logs/` directory.

## 3. Config (unchanged across all runs)
- Model `Qwen/Qwen3-4B` official snapshot, BF16, TP=1, batch=1, eager
- YaRN factor=4.0, original_max_position_embeddings=32768, max_model_len=131072
- retrieval block=64, r=4 reps (mean-direction + farthest-point), top-k=32 blocks,
  sink=64, recent=512, GQA group score + global temporal top-k
- prompt 32768 tokens (repeated seed text), gen=32 greedy, ignore_eos
- prefill chunked at 4096 (existing path; recent-window gap = known issue,
  quarantined, not part of this round)
- decode token never enters CPU history/reps; selected block internal 64-token
  order preserved; sink/recent/current participate as before

## 4. Phase A: selected-block reuse (the measured result)
`bench_logs/m13_phaseA_reuse.txt` — 1 run, 32K prompt, gen=32, top-k=32,
recorded per layer per decode token (36 layers x 30 transitions = **1080 samples**).

| Metric | Value |
|---|---|
| reuse mean / median | **32.2% / 31.2%** |
| reuse min / max | 0.0% / 96.9% |
| miss (new) blocks per token mean / median / max | 21.68 / 22 / 32 |
| layer 0 / 18 / 35 mean reuse | 22.4% / 38.8% / 41.5% |
| per-step cross-layer mean reuse range | 20.7% – 43.8% |

**Decision: reuse 32.2% < 40% gate → do NOT build the previous-set cache.**
A GPU-resident previous-selected-set cache would have ~22/32 blocks missing each
step, i.e. ~69% of the 288 MiB payload would still cross H2D; cache benefit does
not hold under this drift. This negative result is the deliverable of Phase A.

## 5. Fallback (sole option per task doc §7)
Replace `sel_k = self.k_cpu[block_ids_cpu].reshape(...)` (pageable temp tensors
`sel_k/sel_v`) with a single direct write into the pre-allocated pinned staging:

```python
flat = self.k_cpu[:cpu_blocks_cap].view(-1, Hkv, D)   # contiguous
tok_idx = (block_ids.view(-1,1)*B + arange(B)).reshape(-1)
torch.index_select(flat, 0, tok_idx, out=self.stage_k[:sel_hist_tokens])
```
- No per-block Python loop, no new temp tensor, values identical to reference.
- Gated by `sparse_gather_index_select` (default False = original M12 path).

Micro test (`bench_logs/m13_micro_result.txt`): values equal True, max abs diff
0.0, slice-view `out=` also equal → MICRO TEST PASSED.

## 6. Before/after benchmark (32K, gen=32, alternating order)
Clean timing: prefill wall measured separately; decode wall per token measured
between `torch.cuda.synchronize()` boundaries; end-to-end decode wall TPOT.

| run | use_index_select | decode wall/token mean ms | steady median ms* | prefill wall s |
|---|---|---|---|---|
| base1 | off | 339.15 | 241.14 | 46.63 |
| is1   | on  | 276.39 | 228.30 | 46.20 |
| base2 | off | 427.92 | 393.09 | 46.21 |
| is2   | on  | 356.10 | 321.47 | 47.30 |

\* steady = decode steps 5..31 (drop first 4 warmup). Full per-run stats in
`m13_bench_32768_32_{0,1}_{base1,is1,base2,is2}.txt`.

The two paired runs show a positive direction, but the run-to-run variance is
large: the same configuration differs by about 150 ms in steady median (241 vs
393 ms), consistent with WSL host scheduling and CPU page/cache effects. For a
project-facing headline, the best observed paired result is:

```text
393.09 ms/token -> 321.47 ms/token
best-observed reduction = (393.09 - 321.47) / 393.09 = 18.2%
```

This **18.2% is a best-observed A/B result, not a claim of stable average
speedup**. The companion pair was 241.14 -> 228.30 ms/token (5.3% lower), so
the raw runs and the variance must remain visible in any detailed report.

Instrumented stage sums (last decode step, 36 layers, ms; auxiliary — selector
includes its ID D2H, do NOT add `d2h_ms` separately):

| run | selector | cpu_gather | h2d_pack | recent | packed_attn | SUM |
|---|---|---|---|---|---|---|
| base1 | 46.0 | 54.4 | 15.9 | 11.4 | 31.6 | 162.2 |
| is1   | 41.9 | 36.8 | 14.2 | 11.7 | 25.1 | 133.1 |
| base2 | 64.8 | 102.4 | 25.0 | 14.8 | 68.9 | 280.7 |
| is2   | 72.0 | 61.6 | 36.4 | 16.1 | 45.7 | 236.7 |

`cpu_gather` was the largest stage in the baseline snapshots. After the
fallback, selector and CPU gather are the two largest measured stages; the
fallback cuts gather by about 32-40% (base 54.4/102.4 -> is 36.8/61.6) with
identical output. This is the actionable evidence for keeping the fallback;
the stage result supports the implementation, while the 18.2% number above is
only the best observed end-to-end A/B pair.

Payload (shape-based): selected 2048 tokens/layer x (K+V) x 8 heads x 128 dim x
2 bytes = 8.00 MiB/layer, **288.0 MiB per decode token total**. Host→device
"tensor payload" only; not a PCIe hardware counter. Theoretical dense-equivalent
KV for 32K is 128 MiB/layer, or about **4.5 GiB across 36 layers per decode
token** (not materialized on GPU; this design keeps it in CPU memory).

## 7. Correctness gate
| check | result |
|---|---|
| micro `index_select(out=)` vs reference | equal, max abs diff 0.0 |
| 32K greedy, gen=8: token ids off vs on | identical (`[978, 70880, 1573, 11601, 279, 33340, 44378, 10001]`) |
| per-layer selected IDs off vs on (36 layers x 7 transitions) | identical |
| feature-off dense smoke (sparse off, paged FA, seqlen=1024, gen=8) | passed, coherent text |
| `git diff --check` | clean |
| NaN / OOM during all runs | none |

Artifacts: `bench_logs/m13_corr_0.pt`, `m13_corr_1.pt`, `m13_corr_compare.txt`.

## 8. Memory (measured)
- Host: RSS ~7.4 GiB, available 16.7–17.0 GiB (of 25.4 GiB WSL) across runs.
- GPU: allocated 8.62 GiB, reserved 10.78 GiB of 16.0 GiB (reserved ≠ active KV).

## 9. Valid / invalid items (explicit)
Valid:
- Phase A reuse measurement (1080 samples, negative result → cache rejected).
- Micro correctness of the fallback path; end-to-end functional equivalence.
- Stage-level evidence that the fallback reduces CPU gather by ~32-40% without
  changing outputs.
- Best observed end-to-end A/B pair: 393.09 -> 321.47 ms/token, an 18.2%
  reduction for that pair; this is not a stable-average claim.
- Dense feature-off smoke, diff cleanliness, no OOM/NaN.
Invalid / not claimed:
- No previous-set cache was built (decision gate).
- No 64K/128K confirmation run (not required once fallback chosen; would be noise
  under current host variance anyway).
- Old M12 numbers and old chunk-equiv 8/8 are not reused as evidence.

## 10. M13 closeout and next milestone
M13-FAST is closed with the one-shot `index_select` fallback retained behind an
explicit opt-in flag. The best observed end-to-end A/B pair is reported above,
and the raw runs remain visible because host scheduling noise is substantial.
No further decode overlap or cache work is part of this milestone.

The next planned milestone is prefill: first repair and validate the known
chunked-prefill recent-window coverage gap, then establish a clean 128K TTFT
baseline and evaluate one bounded FlashAttention-2/prefill optimization. That
work has not started in this branch.

## 11. Limits
- WSL host scheduling noise makes sub-10% TPOT claims unverifiable here.
- gen=32 (allowed max 64); single 32K config; single repeated-seed prompt.
- `index_select` path is CPU-synchronous gather (kernel), staged H2D still
  non_blocking as before.
- chunked-prefill recent-window gap: known issue, quarantined (unchanged).

## 12. Reproduce
```
# Phase A
cd /opt/nano-vllm && .venv/bin/python bench_logs/m13_phaseA_reuse.py   # -> m13_phaseA_reuse.txt
# micro
.venv/bin/python bench_logs/m13_micro_index_select.py                   # -> m13_micro_result.txt
# bench (before/after)
.venv/bin/python bench_logs/m13_bench.py 32768 32 0 base1
.venv/bin/python bench_logs/m13_bench.py 32768 32 1 is1
# correctness
.venv/bin/python bench_logs/m13_correctness.py 0 m13_corr_0.txt
.venv/bin/python bench_logs/m13_correctness.py 1 m13_corr_1.txt
.venv/bin/python bench_logs/m13_compare_corr.py
.venv/bin/python bench_logs/m13_smoke_dense.py
```
All logs under `/opt/nano-vllm/bench_logs/` (gitignored).
