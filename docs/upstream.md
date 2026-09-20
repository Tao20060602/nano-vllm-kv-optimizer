# Upstream baseline

NanoKV is built as an extension of the public [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) project. The imported upstream snapshot is pinned to:

- Repository: `https://github.com/GeeeekExplorer/nano-vllm`
- Commit: `bb823b3e06983d71485a8e1f23715ebd87d98ef8`
- Import date: `2026-09-19`
- License: MIT (see [`LICENSE`](../LICENSE))

The upstream files are retained as the execution baseline. NanoKV must preserve the upstream attribution and must not describe the upstream engine as a new implementation from scratch.

## Baseline commands

From the repository root, after installing the upstream dependencies and placing a supported Qwen3-0.6B model at `~/huggingface/Qwen3-0.6B/`:

```bash
python example.py
python bench.py
```

For a deterministic, machine-readable environment/baseline probe:

```bash
python benchmarks/baseline.py --output benchmarks/results/baseline_environment.json
```

The probe records Python, PyTorch, CUDA, GPU, dependency, model-path, and Git state. It never substitutes estimated performance numbers for a run that could not execute.

## Current machine evidence

On the machine used to initialize this workspace (2026-09-19), the default
PowerShell `PATH` probe found:

- `python`: not available on `PATH`;
- the default `~/huggingface/Qwen3-0.6B` model directory: absent;
- therefore the upstream example and benchmark have not been run here;
- GPU correctness and latency remain pending until a Python/PyTorch/CUDA environment and model are provided.

On 2026-09-20, an existing D-drive site-packages directory was also found at
`D:\\music\\venv\\Lib\\site-packages`. It contains CPU-only PyTorch 2.14.0 and
NumPy. With the repository and that directory on `PYTHONPATH`, the pure Python
Milestone 1 tests pass (`4 passed`); this does not provide CUDA or upstream model
coverage. The upstream example/benchmark still require Transformers, Triton,
FlashAttention, a CUDA build, and model weights.

The physical machine does have an NVIDIA GPU: `nvidia-smi` reports an NVIDIA
GeForce RTX 3080 with 16 GB and driver `616.92` (CUDA UMD `13.4`). The remaining
gap is therefore the runnable Python/CUDA package stack, not GPU hardware.

Re-run the probe and the two upstream commands after installing the environment. Do not mark the GPU portions of the project complete until their real output is saved under `benchmarks/results/`.
