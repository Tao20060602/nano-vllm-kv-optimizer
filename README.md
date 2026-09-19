<p align="center">
<img width="300" src="assets/logo.png">
</p>

# nano-vLLM KV Optimizer

An experimental extension of [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) focused on KV-cache-aware scheduling, memory management, and multi-GPU inference optimization.

The project keeps nano-vLLM's small, readable codebase while adding the observability and benchmarks needed to evaluate scheduling and KV cache changes rigorously.

## Project Goals

- Measure TTFT, TPOT, end-to-end latency, and phase throughput.
- Build reproducible mixed-workload benchmarks.
- Implement latency-aware mixed prefill/decode scheduling.
- Add cost-aware preemption and prefix-cache eviction policies.
- Explore KV-aware request routing across multiple GPU workers.
- Profile CUDA kernels, memory traffic, and NCCL communication.

See [the roadmap](docs/ROADMAP.md) for milestones and success criteria.

## Current Status

- [x] Fork baseline pinned to upstream commit `bb823b3`.
- [x] Request-level TTFT, TPOT, and end-to-end metrics.
- [x] Step-level prefill/decode throughput metrics.
- [x] Reproducible baseline benchmark output in JSON and CSV.
- [ ] Latency-aware mixed prefill/decode scheduler.
- [ ] Cost-aware preemption.
- [ ] Prefix-cache LRU/LFU policy and cache telemetry.
- [ ] Multi-GPU KV-aware request routing.

## Installation

nano-vLLM requires Linux, an NVIDIA GPU, and Python 3.10–3.12.

```bash
git clone https://github.com/Tao20060602/nano-vllm-kv-optimizer.git
cd nano-vllm-kv-optimizer
pip install -e .
```

Download a small Qwen3 checkpoint for development:

```bash
huggingface-cli download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/
```

## Quick Start

```python
from nanovllm import LLM, SamplingParams

llm = LLM(
    "~/huggingface/Qwen3-0.6B",
    enforce_eager=True,
    tensor_parallel_size=1,
)

outputs = llm.generate(
    ["Explain paged KV cache."],
    SamplingParams(temperature=0.6, max_tokens=128),
)

print(outputs[0]["text"])
print(llm.get_metrics()["summary"])
```

## Baseline Benchmark

```bash
python benchmarks/baseline.py \
  --model ~/huggingface/Qwen3-0.6B \
  --num-requests 64 \
  --min-input-length 64 \
  --max-input-length 512 \
  --min-output-length 32 \
  --max-output-length 256 \
  --output-dir benchmark-results/baseline
```

The benchmark writes:

```text
summary.json       Aggregate TTFT, TPOT, E2E, and throughput
requests.csv       Per-request latency and token counts
steps.csv          Per-step phase, batch size, latency, and throughput
```

Benchmark methodology is documented in [docs/BENCHMARKING.md](docs/BENCHMARKING.md).

## Upstream

This project is based on the MIT-licensed [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm). The local Git configuration uses:

```text
origin   https://github.com/Tao20060602/nano-vllm-kv-optimizer.git
upstream https://github.com/GeeeekExplorer/nano-vllm.git
```

To sync upstream changes:

```bash
git fetch upstream
git rebase upstream/main
```

## License

MIT. See [LICENSE](LICENSE).
