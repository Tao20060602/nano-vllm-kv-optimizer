# Authoritative nano-vLLM environment

This document is the persistent environment contract for NanoKV work. It exists so a new Codex conversation does not rediscover the wrong Windows Python or the wrong WSL distribution.

## Native Linux continuation (planned, not yet verified)

The user plans to resume on native Linux. On the current Windows machine,
the WSL routing below remains mandatory. On an actual native Linux machine,
use the user-selected clone and a verified Linux venv directly; do not run
`wsl.exe`, the PowerShell wrapper, or assume `/opt/nano-vllm` is present.

Read the [operator plan and Linux handoff](SEGMENTED_GQA_PREFILL_PLAN_AND_LINUX_HANDOFF.md)
before setup changes. Verify and record OS, checkout SHA, GPU/driver, Python,
PyTorch CUDA runtime, toolkit, Triton, FlashAttention and model paths. The
table below is the old verified reference, not proof of a new installation.
Models, HF symlink targets and ignored logs/traces require separate transfer;
do not copy a venv as the new environment or remove WSL before verification.

Recent system comparisons use Qwen3-4B snapshot
`1cfa9a7208912126459214e8b04321603b3df60c`; Qwen3-0.6B below remains the
old environment-check model. Pin the same 4B model for comparable system
A/B runs. Operator microbenchmarks do not require a model.

## Source of truth

| Item | Value |
| --- | --- |
| WSL distribution | `NanoVLLM-Ubuntu` |
| OS | Ubuntu 24.04 LTS |
| WSL virtual disk | `D:\AI\WSL\Ubuntu24.04-admin\ext4.vhdx` |
| Authoritative repository | `/opt/nano-vllm` |
| Virtual environment | `/opt/nano-vllm/.venv` |
| Python | `/usr/bin/python3.12` |
| uv | `/usr/local/bin/uv` |
| CUDA home | `/usr/local/cuda-12.8` |
| Model | `/opt/models/Qwen3-0.6B` |
| Hugging Face cache | `/opt/models/.cache/huggingface` |
| GPU | NVIDIA GeForce RTX 3080 Laptop GPU, 16 GB |
| Compute capability | 8.6 |
| PyTorch | 2.7.1+cu128 |
| Triton | 3.3.1 |
| FlashAttention | 2.8.3.post1 |
| nano-vLLM | 0.2.0, upstream commit `bb823b3` |

The environment has previously been verified with `torch.cuda.is_available() == True`, and nano-vLLM successfully generated text with the local Qwen3-0.6B model.

## Why Windows is not a second runtime

The `ext4.vhdx` file is the backing disk for WSL, not a Windows directory containing `/opt`. Windows tools cannot discover `/opt/nano-vllm/.venv/bin/python` by recursively searching drive D. A Windows Python probe therefore says nothing about the WSL environment.

This project depends on the Linux CUDA ecosystem, including Triton, FlashAttention, and NCCL-oriented execution. Maintaining a second native Windows environment would duplicate several gigabytes of packages and model data while introducing a different compatibility surface. Windows is the control plane; WSL is the development and execution plane.

## Canonical invocation

From this Windows coordination repository:

```powershell
.\scripts\wsl-nanovllm.ps1 -Command "python --version"
.\scripts\wsl-nanovllm.ps1 -Command "python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'"
.\scripts\wsl-nanovllm.ps1 -Command "git status --short --branch"
```

The wrapper always selects `NanoVLLM-Ubuntu`, changes to `/opt/nano-vllm`, activates `.venv`, sets `CUDA_HOME` and `HF_HOME`, and then runs the supplied command.

## New-conversation behavior

Codex reads repository-level `AGENTS.md` when a task starts in this repository. A narrow global rule is also installed at `%USERPROFILE%\.codex\AGENTS.md`, so projectless/new conversations that mention NanoKV, nano-vLLM, or AlayaDB are redirected here instead of probing Windows first.

The rule is reliable when either:

1. the task is opened from this repository or `/opt/nano-vllm`; or
2. the task clearly identifies itself as NanoKV/nano-vLLM/AlayaDB work.

No instruction file can identify an unrelated, unnamed project with perfect certainty. When starting a new task manually, selecting this project remains the strongest signal.

## Troubleshooting

- If a command enters the wrong distro, verify it contains `-d NanoVLLM-Ubuntu`.
- If Python resolves to a Windows path, use the wrapper or `/opt/nano-vllm/.venv/bin/python` inside WSL.
- If the model is reported missing, check `/opt/models/Qwen3-0.6B`, not a Windows Hugging Face directory.
- If an agent proposes installing Windows CUDA packages, point it to `AGENTS.md` and stop that installation.
- If instructions appear stale, start a new task from this repository; Codex loads `AGENTS.md` when the task/session starts.
