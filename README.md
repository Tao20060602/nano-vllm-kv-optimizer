<p align="center">
<img width="300" src="assets/logo.png">
</p>

# NanoKV

**NanoKV is an educational, single-sequence sparse-attention prototype built
on [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm).** Its current
Qwen3-4B path keeps full per-layer K/V history on the CPU, GPU-side block
representatives plus protected sink/recent K/V, and gathers selected history
for decode. It is a teaching and measurement project, not a production serving
stack.

## Native Linux operator experiment (2026-10-07)

An opt-in bridge now connects the independent segmented attention kernels to
M12 prefill, with shared scratch and cached dynamic pointer binding. Nine fresh
processes compared the original FA2, reusable FA2 and operator paths on two
fixed 16,480-token prompts. Complete prefill timing showed no stable gain;
downstream tail retrieval sets differed despite matching first generated tokens.
The default remains the original `flash` backend. See the
[native integration report](docs/NATIVE_SEGMENTED_ADAPTER_REPORT.md) and
[reproduction guide](docs/NATIVE_SEGMENTED_ADAPTER_USAGE.md). For reading existing results or
checking committed evidence without a GPU, see the [evidence index](docs/REPRODUCTION_INDEX.md).
A concise [portfolio evidence guide](docs/PORTFOLIO_ENGINE_EVIDENCE.md) maps claims
to their measured boundaries.

## Current status — project closeout (2026-10-03)

The current reference setup is Qwen3-4B BF16 (snapshot
`1cfa9a7208912126459214e8b04321603b3df60c`) on an RTX 3080 Laptop (16 GiB),
TP=1/eager, YaRN, 32K repeated-text prompt, 4096-token prefill chunks,
64-token blocks with `r=4` representatives, Top-32 selection, a 64-token sink,
and a 512-token recent window. The reported decode comparisons use greedy
generation. This is a narrow engineering workload, not a general quality or
serving benchmark.

| Path | Current default | Evidence boundary |
| --- | --- | --- |
| Direct pinned CPU gather (`index_select(..., out=pinned)`) | On by default (M18) | Three matched 32K pairs had identical output IDs; paired steady-median reductions were 11.8–18.8% on one repeated prompt. This does not establish a general gain. |
| Static protected-block mask (M21) | Opt-in, off by default | Two fresh-process pairs matched output IDs and selected-block histories; mean decode time after the first four steps was 7.40% and 10.46% lower per pair on that prompt. |
| Selector CUDA graph (M19) | Opt-in, off by default | Corrected fresh-process timing direction was mixed; the interleaved result does not settle the effect. |
| K/V copy pipeline (M20) | Opt-in, off by default | All three latest decode comparisons recorded higher pipeline times; they do not establish a general effect or prove a causal slowdown. |
| Adaptive decode or reduced prefill Top-K (M16–M17) | Experimental, off by default | Adaptive decode did not show a TPOT gain. A small multi-key screen regressed when prefill K was lowered, so quality preservation is not established. |

Experiment details: [M16](docs/m16_dynamic_topk_results.md),
[M17](docs/m17_prefill_budget_results.md),
[M18](docs/m18_nsight_gather_results.md),
[M19](docs/m19_selector_graph_results.md),
[M20](docs/m20_gather_results.md), and
[M21](docs/m21_selector_static_mask_results.md).

For the full Chinese decision history—from prefix reuse and graph retrieval
to the final optimizations, failed candidates, quality gaps and handoff—read
[project history and handoff](docs/NANOKV_PROJECT_HISTORY_AND_HANDOFF.md).

### Final quality evidence (M22)

The pinned NVIDIA/RULER generators and reference-matching metric were used on
80 fixed prompts (220 generations). This is a bounded subset, not a full
leaderboard or broad long-context quality claim.

| Context / task | Samples | Dense | Sparse baseline | M21 |
| --- | ---: | ---: | ---: | ---: |
| 8K single target | 20 | 100 | 100 | 100 |
| 8K similar-key distractors | 20 | 100 | 80 | 80 |
| 8K variable tracking | 20 | 97 | 92 | 92 |
| 32K single target | 10 | not run | 100 | 100 |
| 32K similar-key distractors | 10 | not run | 0 | 0 |

M21 and the sparse baseline produced identical generated token IDs and text
on **80/80** prompts. However, the sparse configuration loses quality relative
to dense in the 8K distractor and variable-tracking tasks, and fails all ten
32K distractor cases. **This is not lossless sparse attention or validated
general long-context retrieval.** Variable tracking reports reference-item
recall, not the percentage of fully solved chains. All generations reached
their fixed task cap. 32K dense was not run; no dense-paired conclusion is made
at that length.

See the [complete quality report](docs/m22_quality_closeout_results.md),
[public per-sample evidence](benchmarks/results/m22_quality/20261003-closeout/),
and [project closeout / reproduction boundaries](docs/PROJECT_CLOSEOUT.md).

**Project status:** the M22 system baseline is closed. On 2026-10-05 the user
selected a bounded follow-on: an independently benchmarked segmented-KV GQA
prefill operator, with NanoKV integration only if evidence supports it.
This is a plan, not an implemented kernel or a new acceleration result; the
existing retrieval/quality limitations remain unchanged.

### Next direction and native Linux handoff (planning only)

Read the [operator plan and Linux handoff](docs/SEGMENTED_GQA_PREFILL_PLAN_AND_LINUX_HANDOFF.md)
for the source-audited input layout, old Nsight evidence, strong FA2 baselines,
correctness contract, staged TODOs and stop conditions. The first engineering
task is a cost baseline, not a full CUDA implementation. Native Linux setup
has not yet been verified. Models, virtual environments and ignored traces
do not come with a GitHub clone; preserve them separately before retiring WSL.

## Attribution and scope

NanoKV builds on the public nano-vLLM project, imported at upstream commit
`bb823b3e06983d71485a8e1f23715ebd87d98ef8` (MIT license, 2026-09-19). The
upstream engine, model definitions, attention kernels, and sampler are
retained as the execution baseline. NanoKV adds sparse-attention experiments
and instrumentation on top; it does **not** claim to have reimplemented
nano-vLLM from scratch, and it is **not** a complete AlayaDB replacement.

See [`docs/upstream.md`](docs/upstream.md) for the exact upstream pin and
[`docs/architecture.md`](docs/architecture.md) for the baseline call chain
that must be preserved when the feature flag is off.

## Historical M0–M6: CPU-backed prefix reuse

The following sections preserve the earlier CPUBlockStore prefix-reuse
milestones. They describe the M0–M6 implementation and measurements, not the
current Qwen3-4B sparse-decode configuration above.

### What NanoKV added over upstream

Upstream nano-vLLM has GPU-resident prefix caching: when a physical GPU
block's reference count drops to zero, its hash metadata is dropped and the
block is reused. There is no persistent home for a block that has been
evicted from GPU.

NanoKV adds:

- an independent `CPUBlockStore` abstraction (pageable and pinned host memory);
- a `ContextDB` that does longest-prefix matching over both GPU and CPU tiers;
- a cache fingerprint that gates reuse by model, dtype, layer count, KV-head
  layout, block size, and TP size;
- synchronous CPU -> GPU restoration of matched full blocks, with a
  transactional fallback to full prefill on any copy failure;
- LRU eviction with a fixed block capacity;
- structured, per-request metrics: lookup time, H2D load time and bytes,
  prefill time, reused/prefilled token counts, TTFT;
- a reproducible benchmark comparing cold, GPU-hit, CPU-pageable,
  CPU-pinned, and partial-hit behavior.

### M0–M6 system architecture

```text
LLMEngine.add_request(token IDs)
  -> ContextDB.longest_prefix_lookup()
       -> GPU block index (upstream hash chain)
       -> CPU  block index (token-hash, collision-checked)
  -> Scheduler chooses the longer match (GPU wins ties)
  -> if CPU hit:
       BlockManager allocates fresh GPU physical blocks
       ModelRunner synchronously loads matched full blocks CPU -> GPU
       BlockManager registers restored block hashes only after all loads succeed
  -> ModelRunner prefills only the suffix / non-aligned tail
  -> Sampler -> first token (TTFT boundary)
  -> decode loop
  -> on completion: completed full prompt blocks are persisted GPU -> CPU
     (with LRU eviction if CPU store is at capacity)
```

`BlockManager` owns logical block tables, refcounts, and the GPU hash index.
`ModelRunner` owns `GPUBlockStore`, `CPUBlockStore`, and all KV tensor copies.
The scheduler never touches K/V tensors directly.

### M0–M6 quick start

The authoritative environment is WSL2 distribution `NanoVLLM-Ubuntu`,
repo `/opt/nano-vllm`, virtualenv `.venv`:

```bash
wsl.exe -d NanoVLLM-Ubuntu
cd /opt/nano-vllm
source .venv/bin/activate
export CUDA_HOME=/usr/local/cuda-12.8
export HF_HOME=/opt/models/.cache/huggingface

# upstream baseline (flags off)
python example.py

# CPU-backed prefix reuse
python - <<'PY'
from nanovllm import LLM, SamplingParams

llm = LLM(
    "/opt/models/Qwen3-0.6B",
    enforce_eager=True,
    tensor_parallel_size=1,
    enable_cache_metrics=True,
    enable_cpu_cache=True,
    cpu_cache_pinned=True,
)
sp = SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True)

llm.generate([list(range(2048))], sp)         # primes GPU + CPU store
llm.clear_gpu_prefix_cache()                   # forces next hit to come from CPU
out = llm.generate([list(range(2048))], sp)
print(llm.get_cache_metrics())
print(llm.get_cpu_cache_stats())
PY
```

### M0–M6 feature matrix

| Capability | Status |
| --- | --- |
| GPU prefix caching (upstream) | retained, unchanged when flag off |
| Longest-prefix lookup (GPU + CPU) | implemented |
| Hash-collision-safe matching (token-id verification) | implemented |
| Non-aligned tail never reused | enforced |
| CPU KV store, pageable and pinned | implemented |
| LRU eviction with block capacity | implemented |
| GPU eviction -> CPU restore | implemented |
| Restore failure -> full prefill fallback | implemented |
| Cache fingerprint gating (model/dtype/layout) | implemented |
| Per-request metrics (lookup/H2D/prefill/TTFT) | implemented |
| 28 unit + integration tests | passing |
| Sparse attention | not implemented |
| ANN / vector index lookup | not implemented |
| Async / overlapped transfer | not implemented |
| Distributed serving, TP>1, batch>1 | not implemented |
| SSD / remote tier, compression, quantization | not implemented |

### M0–M6 correctness snapshot

See [`docs/correctness.md`](docs/correctness.md) for the full evidence list.
Summary:

- 28 tests pass; every M5 end-to-end case compares generated greedy token IDs
  against an independently cold prompt and they match.
- A restored CPU path transfers exactly the expected bytes (e.g. two blocks =
  58,720,256 B) and leaves GPU refcounts at zero afterward.
- An injected H2D failure falls back to a full prefill and still produces the
  cold-baseline tokens.
- With `enable_cpu_cache=False`, the engine follows the upstream call chain.

### Historical benchmark results (M6)

Full methodology: [`docs/benchmark_methodology.md`](docs/benchmark_methodology.md).
Raw data and generated report: `benchmarks/results/m6/`.

Configuration: Qwen3-0.6B, block size 256, 5 warmups + 20 measurements per
point, `torch.cuda.synchronize()` around every call. TTFT p50 in ms:

| Prefix | cold | GPU hit | CPU pageable | CPU pinned |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 39.3 | 61.9 | 46.7 | 47.1 |
| 512 | 37.7 | 64.6 | 53.4 | 48.3 |
| 1024 | 47.8 | 80.2 | 66.9 | 59.1 |
| 2048 | 91.0 | 53.5 | 88.3 | 83.5 |
| 4096 | 192.9 | 58.7 | 132.6 | 94.8 |

Partial-reuse (2064-token prompt, pinned CPU): TTFT p50 improves roughly
monotonically as the reused fraction rises (25%: 110 ms, 50%: 89 ms, 75%:
67 ms, 100%: 69 ms).

Takeaways from the generated `conclusions.json`:

- Break-even (CPU reuse beats cold prefill) starts at **2048 tokens** on this
  GPU; at shorter prefixes the H2D transfer cost exceeds the prefill saved.
- Pinned memory beats pageable memory most at long prefixes (~28% faster at
  4096 tokens) and is close at short prefixes.
- The CPU prefix lookup itself is <0.5% of TTFT; the measured cost is the
  H2D restore, not the index.
- CPU store capacity in the benchmark is 64 blocks = 16,384 cached tokens.

### M0–M6 hardware and software environment

- WSL2 (`Linux 6.18.33.2-microsoft-standard-WSL2`), glibc 2.39
- GPU: NVIDIA GeForce RTX 3080 Laptop (16 GiB)
- PyTorch 2.7.1+cu128, CUDA 12.8 build, Python 3.12
- Model: Qwen3-0.6B, bfloat16, block size 256
- One KV block = 29,360,128 bytes (28 MiB) for this model

### M0–M6 design choices

- **Upstream stays the default.** Every NanoKV behavior is behind
  `enable_cpu_cache` / `enable_reusable_cache` flags that default off.
- **Block-level, not token-level, offload.** Only complete 256-token blocks
  are stored to and restored from CPU; the tail is always recomputed, which
  keeps the copy granularity aligned with FlashAttention's paged layout.
- **Token-id verification on every hash hit.** A chained xxHash is only a
  first filter; the stored token sequence is compared before reuse, so hash
  collisions cannot corrupt KV.
- **Restore is transactional.** GPU blocks are registered only after all H2D
  copies for that request have succeeded; any error deallocates and reschedules.
- **Per-request metric window.** Metrics are reset at the start of each
  `generate()` so TTFT, lookup, load, and prefill counters describe exactly
  one request.

## Known limitations

See [`docs/limitations.md`](docs/limitations.md) for the current validation
boundary and the separately labeled M0–M6 CPUBlockStore caveats. NanoKV is
single-sequence and educational; the recent timing and output-equivalence
checks do not establish broad quality or production behavior.

## Roadmap

The M22 system baseline remains closed. The user-selected follow-on is the
[segmented GQA prefill operator plan](docs/SEGMENTED_GQA_PREFILL_PLAN_AND_LINUX_HANDOFF.md),
starting with a strong-baseline cost study on a verified Linux environment.
No implementation or new speedup is claimed, and the previous retrieval,
offload and quality backlogs are not automatically reopened.

## Repository layout

```
nanovllm/sparse/       current representative selector and sparse decode path
nanovllm/kvdb/         retained CPU-backed prefix-reuse implementation
nanovllm/engine/       llm_engine, scheduler, model_runner, block_manager
benchmarks/            current sparse experiments and historical M6 drivers
docs/                  architecture, limitations, M13–M22 evidence and decision history
tests/                 correctness and regression tests
```
## Historical milestones M8–M11: block-sparse decode + CPU KV offload

This section preserves the early Qwen3-0.6B milestone snapshot. Its timings
and implementation details are historical; use the M16–M21 reports and the
completed M22 closeout report for later evidence.

M8–M11 extend the engine with block-level sparse attention inspired by AlayaDB
DIPR/DIPRS. This is a **single-sequence educational prototype**:

- exact Block-DIPR oracle (M8), real-model trace + synchronous CPU KV offload
  (M9), sampled-query-guided block graph with DIPRS traversal (M10), and
  engine integration where packed sparse attention actually feeds token
  generation (M11).

**Project positioning:** NanoKV is an educational block-level sparse-attention
and CPU-KV-offload prototype inspired by AlayaDB DIPR/DIPRS. It uses exact
scanning as an oracle and a *simplified sampled-query-guided graph*, not a
reproduction of AlayaDB's production RoarGraph.

### Data flow (sparse decode)

```
dense FlashAttention prefill (GPU)
  -> each layer copies post-RoPE K/V to CPU history
  -> build frozen reps + per-KV-head block graph from sampled (non-final) q
decode token:
  append current post-RoPE K/V to CPU
  -> CPU selector (full/exact/top-k/mean/real/KNN/qg)
  -> force first + recent windows
  -> gather selected rows to pinned staging
  -> H2D only packed K/V
  -> packed PyTorch attention -> o_proj -> sampler
```

### Commands

```bash
# feature-off: original paged FlashAttention (unchanged)
python -m nanovllm --model /opt/models/Qwen3-0.6B

# one-command sparse demo (dense baseline vs query-guided)
bash scripts/run_m11_demo.sh

# full engine benchmark (writes JSON/CSV + plots)
.venv/bin/python benchmarks/benchmark_engine_sparse.py
.venv/bin/python benchmarks/plot_m11_results.py
```

### Hardware

RTX 3080 Laptop (16 GB), CUDA 12.8, torch 2.7.1+cu128, Qwen3-0.6B bf16, WSL2.

### M11 result summary (Qwen3-0.6B, exactly-2048-token prompt, greedy, 16 tokens)

| selector | selected ratio | pre-window recall | agreement vs dense | p50 TPOT |
|---|---|---|---|---|
| dense FlashAttention (feature-off) | 1.0 | n/a | 1.00 | ~35 ms |
| full CPU offload | 1.00 | 1.00 | 1.00 | 508 ms |
| exact Block-DIPR | 1.00 | 1.00 | 1.00 | 643 ms |
| top_k | 0.91 | 0.91 | 1.00 | 747 ms |
| mean representative | 0.97 | ~1.0 | 1.00 | 938 ms |
| r=4 real representative | 0.41 | ~1.0 | 1.00 | 529 ms |
| KNN graph | 0.38 | ~1.0 | 1.00 | 1110 ms |
| query-guided graph | 0.38 | ~1.0 | 1.00 | 1299 ms |

Counters: 28 layer prefill inits, 15 generated steps, 420 decode layer calls,
0 dense fallbacks. beta_raw=48, beta_scaled_logit=4.24. Sparse mode allocates
**0** paged GPU KV blocks; the pageable CPU history buffer is ~226 MiB over 28
layers at this 2068-token capacity (a linear ~896 MiB only at an 8192-token
capacity). Packed GPU attention is sub-millisecond; the higher TPOT comes from
synchronous Python CPU selection over a 2048-token history, **not** attention.
No end-to-end speedup is claimed.

The M11 dense reference is the **feature-off, real paged FlashAttention engine**
(upstream kernel), not the M9/M10 `dense_decode_attention` PyTorch replay. The
packed GPU attention used in sparse decode is a PyTorch implementation. See
`docs/nanokv_m11_results.md` and `docs/nanokv_interview_guide.md`.
