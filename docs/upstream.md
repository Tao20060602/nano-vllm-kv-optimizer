# Upstream baseline

NanoKV is built as an extension of the public [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) project. The imported upstream snapshot is pinned to:

- Repository: `https://github.com/GeeeekExplorer/nano-vllm`
- Commit: `bb823b3e06983d71485a8e1f23715ebd87d98ef8`
- Import date: `2026-09-19`
- License: MIT (see [`LICENSE`](../LICENSE))

The upstream files are retained as the execution baseline. NanoKV must preserve the upstream attribution and must not describe the upstream engine as a new implementation from scratch.

## Baseline commands

Run from `/opt/nano-vllm` in the `NanoVLLM-Ubuntu` distribution:

```bash
source .venv/bin/activate
export CUDA_HOME=/usr/local/cuda-12.8
export HF_HOME=/opt/models/.cache/huggingface
export NANOVLLM_MODEL=/opt/models/Qwen3-0.6B

python benchmarks/baseline.py \
  --output benchmarks/results/baseline_environment.json
python example.py \
  --output benchmarks/results/upstream_example.json
python bench.py \
  --output benchmarks/results/upstream_benchmark.json
python benchmarks/deterministic_baseline.py \
  --max-tokens 8 \
  --output benchmarks/results/deterministic_baseline.json
```

`example.py` and `bench.py` retain their upstream defaults. The only changes in
Milestone 0 are an explicit `--model`/`NANOVLLM_MODEL` override and optional JSON
output, so the model does not have to be copied or symlinked into the upstream
default directory. The deterministic baseline uses explicit input token IDs,
greedy decoding (`temperature=0`), `ignore_eos=True`, and a fixed output length.

## Verified results

The commands above were run successfully on 2026-09-20 in the authoritative
WSL environment. Direct machine-readable evidence is stored in:

- [`baseline_environment.json`](../benchmarks/results/baseline_environment.json)
- [`upstream_example.json`](../benchmarks/results/upstream_example.json)
- [`upstream_benchmark.json`](../benchmarks/results/upstream_benchmark.json)
- [`deterministic_baseline.json`](../benchmarks/results/deterministic_baseline.json)

The environment probe reports Python 3.12.3, PyTorch 2.7.1+cu128, CUDA available,
and one NVIDIA GeForce RTX 3080 Laptop GPU. The official example completed both
prompts. The unchanged default benchmark workload used 256 sequences, maximum
input/output lengths of 1024, and produced 133,966 output tokens in 57.889 seconds
(2,314.18 output tokens/s). This is a single upstream baseline run, not the later
Milestone 6 statistical benchmark.

The deterministic run produced these eight greedy token IDs:

```text
[151667, 198, 32313, 11, 279, 1196, 1101, 4588]
```

The result also records the allocated KV tensor shape and dtype so later cached
and uncached runs can be compared against the same concrete baseline. Elapsed
time includes engine/model initialization and is evidence of reproducibility,
not a TTFT measurement.
