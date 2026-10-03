# NanoKV project closeout

The user requested a final quality-evidence pass, publication to GitHub main,
and an end to further project development. No additional 64K performance
experiment is part of this closeout. Future work requires an explicit request
to reopen the project, for example to reproduce a result for an interview.

## Authoritative environment

- WSL distribution: `NanoVLLM-Ubuntu`; repository: `/opt/nano-vllm`.
- Python: `/opt/nano-vllm/.venv/bin/python`; CUDA: `/usr/local/cuda-12.8`.
- GPU: RTX 3080 Laptop, 16 GiB, SM86.
- Recent comparative experiments use Qwen3-4B BF16, snapshot
  `1cfa9a7208912126459214e8b04321603b3df60c`, under
  `/opt/models/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/`.
- `HF_HOME=/opt/models/.cache/huggingface`.
- The routing documents' Qwen3-0.6B path remains the environment-check model;
  it is not the model behind the recent Qwen3-4B results.

## Current implementation and decisions

The M12 path stores full per-layer post-RoPE K/V on CPU, scans GPU block
representatives (`r=4`, 64-token blocks), and performs attention on the complete
K/V of selected blocks plus sink/recent context. The reference setup uses
4096-token prefill chunks, fixed Top-32, sink 64 and recent 512.

- Graph retrieval was not retained as the preferred path after measured
  control-flow/search overhead; flat representative scanning is not claimed
  to universally outperform graph retrieval.
- M14 repaired missing previous-recent K/V in chunked prefill.
- M16 repaired cross-request sparse-state leakage. Adaptive decode Top-K
  remains default-off; reduced payload did not establish a TPOT gain.
- M17 lower prefill budgets remain experimental: a small matched quality
  screen detected a regression; CPU gather improvement did not establish an
  end-to-end prefill improvement.
- M18 direct `index_select` into pinned staging is default-on. Its measured
  decode improvement is limited to the documented paired workload.
- M19 selector CUDA graph is opt-in/default-off; fresh-process timings were
  mixed.
- M20 K/V copy pipeline is opt-in/default-off. Nsight confirms overlap, but
  the clean end-to-end comparisons did not show acceleration.
- M21 static protected-block masking is opt-in/default-off. Two fresh-process
  32K pairs reduced Drop4 mean decode latency by 7.40% and 10.46%, with matched
  generated IDs and selected-block histories. These reductions must not be
  added to M18's percentages or extrapolated to other contexts/workloads.
- M22 closes the quality-evidence gap with 80 fixed RULER-derived prompts
  and 220 generations. M21 matches the sparse baseline's token IDs and text
  on 80/80 prompts. Sparse quality is not lossless: 8K distractor retrieval
  scores 80 versus dense 100; VT item recall scores 92 versus dense 97;
  both sparse arms score zero on ten 32K distractor examples (no dense control
  at 32K). The user requested closeout, so these failures are documented rather
  than followed by another optimization cycle.

## Evidence and navigation

- [Final quality evidence](m22_quality_closeout_results.md): bounded,
  official-generator RULER-derived matched comparisons, not a full leaderboard.
- [M18 gather](m18_nsight_gather_results.md), [M19 graph](m19_selector_graph_results.md),
  [M20 pipeline](m20_gather_results.md), [M21 static mask](m21_selector_static_mask_results.md).
- [Current limitations](limitations.md); older reports remain historical
  records rather than statements of current unresolved bugs.
- Quality inference adapter: `benchmarks/benchmark_m16_ruler.py`.
- Reproduction/scoring driver: `benchmarks/run_m22_quality_closeout.py`.
- Public compact results: `benchmarks/results/m22_quality/`.
- Full local predictions, generated prompts, logs and large Nsight files:
  ignored `bench_logs/`. Do not upload model weights or the whole WSL disk.

## Reopening boundaries

Final verification: all 13 quality arms completed; the CPU artifact audit
checked all 220 generations and reproduced all official scores. The focused
M14/M16/M17/M19/M20/M21/M22 regression selection passed **61 tests in 3.22 s**;
Python compilation and `git diff --check` passed. No new full-repository suite
or additional 64K performance measurement is claimed. Model jobs have ended;
no background project monitor or follow-on optimization is scheduled.

128K execution is not evidence of broad 128K quality. Batch/concurrency
throughput, long decode beyond the recent window, wide model-family coverage
and production serving are not established by this project. Do not silently
promote experimental flags or restart an optimization roadmap. First identify
the interviewer's concrete requirement, choose a narrow reproducible test,
and get authorization for any new work or paid compute.
