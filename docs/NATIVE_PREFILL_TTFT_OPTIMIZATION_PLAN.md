# Native M23: fixed-policy first-token optimization

2026-10-07, recorded before candidate measurement. User selected first-token
latency with existing sparse parameters and unchanged outputs. Representative
K and Q-summary changes may be discussed later as explicit quality/performance
tradeoffs; they are not part of this exact-copy experiment.

## Fixed workload and semantics

Use native Qwen3-4B revision `1cfa9a7208912126459214e8b04321603b3df60c`,
BF16/TP1/eager, original FlashAttention path, block64/r4/Top32, sink64/recent512,
one mean-Q summary, chunk4096, pinned index_select gather. Dynamic budget,
selector Graph/static-mask, decode pipeline and operator adapter remain off.
Keep the previous YaRN factor4 override. Threads8, TF32 disabled.

Freeze archive/code repeated example text at 16480 and 32864 tokens
(four/eight main chunks plus a 96-token tail) from the audited operator bridge
prompt units and the pinned tokenizer. These are deterministic engineering
inputs, not production traffic or a quality benchmark. Record complete IDs,
construction, tokenizer/model manifest hashes and input SHA256.

Primary metric: encoded-request enqueue through first generated token, with
synchronized engine steps and no profile, state hashes, hooks or file writes
inside the measured request. Model loading, tokenizer work, allocated-buffer
warm requests and explicit sparse_reset are excluded and recorded separately.
Secondary metric: synchronized step sum/main/tail. This is a single-request
warm engine TTFT, not network TTFT, model cold startup or serving throughput.

## Diagnose before selecting a candidate

Re-extract CPU annotations and CUDA activities from the preserved October7
Chrome traces; never use their incorrect key_averages range attribution.
CPU inclusive ranges contain GPU waits: selector768ms or store557ms on the
main chunk are not exclusive CPU work or recoverable time. CPU gather was
approximately32ms there. Correlate CUDA submissions and GPU timelines; do not
add overlapping activities to wall time. Nsight Systems may confirm these
relationships in a bounded prefill window, separately from benchmark timing.

The first candidate is justified by a concrete extra copy in `_store_kv`:
GPU tensor -> temporary pageable CPU tensor -> CPU history slice, separately
for K and V. Replace only those two statements with blocking GPU-to-history
`copy_(source, non_blocking=False)`. No arithmetic, representative construction,
selector, ordering, visibility or attention backend changes. Non-contiguous V
is handled by PyTorch logical copy semantics and included in real model audit.
The frozen runtime remains untouched; a benchmark-only instance override binds
an exact copy of the original function with only these two substitutions.
Original and generated function hashes and a reverse-substitution check prove
the experimental source difference.

## Acceptance and bounded scope

1. Two audit processes, baseline and candidate: record all ordered selections
   and logical state after each prefill step, bitwise attention-output hashes,
   full used CPU K/V history and reps/sink/recent hashes at first token. Compare
   16 greedy generated tokens per input, including later decode selections.
   Audit timers are instrumented and excluded from performance conclusions.
2. Three fresh-process pairs, orders A/B, B/A, A/B, four inputs each, same model,
   source and fixture hashes. One largest-shape warm request before measurement
   allocates CPU/GPU scratch and compiles all main/tail shapes. Preserve all raw
   request and step timings, temperature/clocks, peak GPU and host memory.
3. Require exact audit equality and first IDs in every performance pair. A
   meaningful positive result requires TTFT ratio geometric mean <=0.97 over
   the twelve paired inputs, overall gain in each process pair and no per-input
   three-pair geometric mean regression. Smaller/noisy results are recorded
   without claiming a stable engine gain; a later candidate gets a new plan.
4. Do not change the default path or overwrite old M7–M22/bridge results.
   Publish finite scope and failures. Existing sparse-vs-dense quality deficits
   remain; exact-copy equality does not turn sparse attention into lossless
   dense inference. Broader representative/Q changes require a separate quality
   protocol and an explicit user tradeoff decision.
