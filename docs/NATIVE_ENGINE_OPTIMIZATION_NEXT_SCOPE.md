# Deferred NanoKV engine optimization scope

Date: 2026-10-07. Status: **deferred; not complete**.

The independent GPU operator project has priority. Revisit NanoKV engine work
after that project is complete and the user has chosen the target workload.
This note records the intended next scope; it does not authorize or claim an
engine optimization result.

## Objective when resumed

Improve the system path that combines sparse attention with CPU-offloaded KV
history. First use profiling to find the measured end-to-end critical path,
including prefill and data movement. Select an optimization only after that
evidence. Then run a matched, uninstrumented benchmark and require a
reproducible net gain at the engine level before changing any default.

CPU gather is one candidate stage, not an assumed bottleneck. The profile must
distinguish selector work, CPU gather, host-to-device transfer, packing,
attention, KV storage, and any other stage that the captured timeline shows on
the critical path. Overlapping CPU ranges and GPU activities must not be added
as if they were exclusive wall time.

## Evidence carried forward

- The historical M22 system work remains bounded by its original workloads and
  quality screen. M18 direct pinned gather showed a narrow decode benefit on
  one repeated 32K prompt; M21 static masking showed two narrow 32K decode
  results. Neither establishes a prefill gain.
- The 2026-10-07 segmented operator bridge completed its three-arm
  unprofiled prefill suite but did not show stable full-prefill acceleration.
  Its numeric attention checks passed the existing tolerance, while selector
  sets differed in later layers after the 96-token tail chunk. It does not
  establish full model-quality equivalence. Details are in the
  [adapter report](NATIVE_SEGMENTED_ADAPTER_REPORT.md) and
  [paired result summary](../benchmarks/results/operator_bridge/m19-native/summary.json).
- The original [closeout profiling plan](NATIVE_ENGINE_CLOSEOUT_PROFILE_PLAN.md)
  specified traces for `flash`, `flash_reuse`, and `operator`. Only
  `flash` and `flash_reuse` diagnostics were captured, for one archive prompt
  at a 4096-token main chunk and a 96-token tail chunk. The `operator` arm was
  **not profiled**; the planned three-arm profile is incomplete. This does
  not undo the separate unprofiled performance suite, which did run all three
  backends.
- The two completed profile summaries and traces are local ignored artifacts
  under `bench_logs/operator_bridge/closeout-profile-20261007/`. They are
  instrumented diagnostics, not clean performance comparisons, hardware
  counters, or evidence that any one stage is the general bottleneck.

The two local summaries also have a parser limitation: several PyTorch
`key_averages()` user ranges report zero CPU time and a device field that does
not match a CPU/GPU stage attribution. Do not use their `ranges` fields for
that attribution. The original Chrome traces retain CPU `user_annotation`
durations and CUDA kernel/copy activities; re-extract CPU ranges from those
records and validate correlation before drawing a bottleneck conclusion.

The older NanoKV milestone M19 refers to selector CUDA graph. The segmented
operator bridge is a separate effort; use the descriptive name and date rather
than reusing the M19 milestone label.

## Work sequence after the operator project

1. Agree on the target workload and user-facing metric before changing code.
   Pin the native checkout, Qwen3-4B snapshot, prompt IDs, sparse configuration,
   and environment needed for a comparable baseline. Preserve the M22 quality
   findings and do not describe the sparse path as lossless.
2. Capture representative, bounded prefill traces first. Compare CPU range
   timing with GPU kernel and copy timelines, and record both activity overlap
   and engine-step wall time. The existing M12 NVTX regions identify
   `prefill_selector`, `prefill_cpu_gather`, `prefill_h2d_pack`,
   `prefill_attention`, and `prefill_store_kv`; split a range only if the
   timeline leaves a material ambiguity. Profile any decode path only if the
   selected workload requires it.
3. State the bottleneck hypothesis from those measurements and choose one
   bounded optimization to test. Do not begin by changing gather, selector
   policy, retrieval budget, or KV placement based only on older traces or
   isolated kernel timing.
4. Compare the candidate with the frozen baseline in fresh processes using
   identical prompts, model revision, settings, and timing boundaries. Keep
   profiler runs out of the performance result. Report per-prompt and paired
   end-to-end deltas and their spread; include TTFT/prefill wall and decode
   TPOT only where the agreed target workload measures them.
5. Check generated IDs and selected-block histories where exact matching is
   the stated criterion, and run the agreed quality screen. Promote a default
   only if a reproducible end-to-end gain meets the criterion set before the
   experiment without an unacceptable correctness or quality regression. If
   it does not, record the result and stop that candidate; a positive result
   is not assumed.

## Tool and permission boundary

The recorded native environment supports PyTorch CPU/CUDA profiling. The
2026-10-07 plan reports that Nsight Systems was not installed and hardware
counter access was restricted. The available PyTorch trace does not require
root or driver changes. If a future question cannot be answered without
Tensor Core, DRAM, or occupancy counters, identify the exact counter and
measurement first, then request the needed administrator permission. No GPU
profiling or environment change is part of the deferred work in this note.
