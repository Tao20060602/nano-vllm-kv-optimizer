# NanoKV / nano-vLLM working environment

## Host routing and native Linux continuation

- On the current Windows host, the mandatory WSL rules below remain in force:
  use `NanoVLLM-Ubuntu:/opt/nano-vllm`, never Windows Python or generic Ubuntu.
- The user explicitly plans to continue this project on native Linux. On an
  actual native Linux host, work in the user-selected clone of this repository;
  do not invoke `wsl.exe` or assume the old absolute paths exist. This is not
  permission to substitute another runtime while still on Windows.
- Before native Linux environment changes, read `docs/environment.md` and
  `docs/SEGMENTED_GQA_PREFILL_PLAN_AND_LINUX_HANDOFF.md`. First verify host,
  checkout, GPU, Python/venv, dependencies and model locations; report actual
  paths/versions rather than claiming the old WSL configuration is installed.
- The native Linux successor has not yet been validated. Do not delete or
  unregister the old WSL environment as part of setup. Follow the handoff's
  artifact-preservation and staged operator scope.
- Qwen3-0.6B below is the old environment-check model. Recent comparative
  system experiments use the pinned Qwen3-4B snapshot documented in the handoff;
  do not substitute 0.6B for a 4B A/B result without an explicit change of scope.

## Authoritative environment

The environment and mandatory routing rules below describe the verified
Windows/WSL host. On an actual native Linux host, use the host-routing section
above and verify the selected clone and venv instead of enforcing WSL paths.

- The only authoritative development and runtime environment for this project is WSL distribution `NanoVLLM-Ubuntu`.
- The authoritative Git worktree is `/opt/nano-vllm` inside that distribution.
- The Python virtual environment is `/opt/nano-vllm/.venv`.
- The local model is `/opt/models/Qwen3-0.6B`.
- The Hugging Face cache is `/opt/models/.cache/huggingface`.
- CUDA is `/usr/local/cuda-12.8`.
- Expected GPU: NVIDIA GeForce RTX 3080 Laptop GPU, compute capability 8.6, 16 GB.
- Expected verified packages: Python 3.12, PyTorch 2.7.1+cu128, Triton 3.3.1, FlashAttention 2.8.3.post1, nano-vLLM 0.2.0 based on upstream commit `bb823b3`.

## Mandatory routing rules

1. Do not use Windows Python, a Windows virtual environment, or a native Windows CUDA installation for nano-vLLM work.
2. Do not install a duplicate Windows copy of PyTorch, Triton, FlashAttention, CUDA, or model weights for this project.
3. Never infer the Linux filesystem from `D:\AI\WSL\Ubuntu24.04-admin\ext4.vhdx`; that file is only the WSL virtual disk backing store.
4. Never use the generic `Ubuntu` distribution for this project. Always pass `-d NanoVLLM-Ubuntu` explicitly.
5. Run source inspection, Git commands, tests, examples, benchmarks, and package operations inside `/opt/nano-vllm`.
6. Use `/opt/nano-vllm/.venv/bin/python` or activate `/opt/nano-vllm/.venv` before Python commands.
7. Set `HF_HOME=/opt/models/.cache/huggingface` when model/cache lookup matters.
8. Treat this Windows directory as a Codex entry/coordination mirror. Do not assume it is synchronized with `/opt/nano-vllm`; inspect the WSL Git state first and make code changes in the WSL worktree.
9. Before reporting missing Python, CUDA, GPU, packages, or model weights, run the WSL environment check below.

## Required first check in a new task

From PowerShell, run:

```powershell
.\scripts\wsl-nanovllm.ps1 -Command "bash scripts/check-wsl-environment.sh"
```

For arbitrary commands, use:

```powershell
.\scripts\wsl-nanovllm.ps1 -Command "git status --short --branch"
.\scripts\wsl-nanovllm.ps1 -Command "python -m pytest -q"
```

Equivalent direct form:

```powershell
wsl.exe -d NanoVLLM-Ubuntu -- bash -lc \
  "cd /opt/nano-vllm && source .venv/bin/activate && export HF_HOME=/opt/models/.cache/huggingface && <command>"
```

Read `docs/environment.md` before changing environment configuration or diagnosing setup failures.
