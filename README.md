<p align="center">
<img width="300" src="assets/logo.png">
</p>

# NanoKV

**NanoKV is an educational extension of [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm)
that implements and evaluates GPU/CPU KV-cache reuse, longest-prefix matching,
KV offloading to host memory, restoration back to GPU, and an observable
cache lifecycle.** It is a teaching and measurement project, not a production
serving stack.

## Attribution and scope

NanoKV builds on the public nano-vLLM project, imported at upstream commit
`bb823b3e06983d71485a8e1f23715ebd87d98ef8` (MIT license, 2026-09-19). The
upstream engine, model definitions, attention kernels, and sampler are
retained as the execution baseline. NanoKV adds a CPU-backed KV tier and
instrumentation on top; it does **not** claim to have reimplemented
nano-vLLM from scratch, and it is **not** a complete AlayaDB replacement.

See [`docs/upstream.md`](docs/upstream.md) for the exact upstream pin and
[`docs/architecture.md`](docs/architecture.md) for the baseline call chain
that must be preserved when the feature flag is off.

## What NanoKV adds over upstream

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

## System architecture

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

## Quick Start

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

## Feature matrix

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

## Correctness

See [`docs/correctness.md`](docs/correctness.md) for the full evidence list.
Summary:

- 28 tests pass; every M5 end-to-end case compares generated greedy token IDs
  against an independently cold prompt and they match.
- A restored CPU path transfers exactly the expected bytes (e.g. two blocks =
  58,720,256 B) and leaves GPU refcounts at zero afterward.
- An injected H2D failure falls back to a full prefill and still produces the
  cold-baseline tokens.
- With `enable_cpu_cache=False`, the engine follows the upstream call chain.

## Benchmark results (Milestone 6)

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

## Hardware and software environment

- WSL2 (`Linux 6.18.33.2-microsoft-standard-WSL2`), glibc 2.39
- GPU: NVIDIA GeForce RTX 3080 Laptop (16 GiB)
- PyTorch 2.7.1+cu128, CUDA 12.8 build, Python 3.12
- Model: Qwen3-0.6B, bfloat16, block size 256
- One KV block = 29,360,128 bytes (28 MiB) for this model

## Key design choices

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

See [`docs/limitations.md`](docs/limitations.md). In short: synchronous copies
only, process-local host memory, batch size one, TP=1, eager only, no sparse
attention / ANN / distributed tier / compression.

## Roadmap

- async / overlapped H2D transfer with prefill of the suffix;
- continuous batching and chunked-prefill interaction with the CPU tier;
- KV compression or low-rank offload;
- a second GPU / multi-process shared CPU pool;
- decode-time KV offload (not just prefill prefix).

## Repository layout

```
nanovllm/kvdb/         fingerprint, metrics, prefix index, CPU/GPU stores
nanovllm/engine/       llm_engine, scheduler, model_runner, block_manager
benchmarks/            validation + M6 benchmark driver and report generator
docs/                  upstream, architecture, correctness, limitations,
                       benchmark methodology, end-to-end reuse walkthrough
tests/                 28 tests, all passing
```