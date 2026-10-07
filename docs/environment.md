# Authoritative nano-vLLM environment

This document is the persistent environment contract for NanoKV work. It exists so a new Codex conversation does not rediscover the wrong Windows Python or the wrong WSL distribution.

## Native Linux continuation (CUDA and short model inference verified)

2026-10-07 continuation: the native environment also completed bounded Qwen3-4B
16K/32K audits, separate Nsight Systems captures and three paired TTFT process
groups. The current checked kernel is `7.0.0-38-generic`; package/model versions
remain the pinned setup below. See [M23 results and scope](NATIVE_TTFT_DIRECT_STORE_REPORT.md)
and [user-directory Nsight installation](NATIVE_NSIGHT_GUIDE.md). Statements below
about a pending cost baseline describe the October6 setup snapshot.

The native Ubuntu checkout is a separate successor to the older WSL
environment below. Use the native checkout and its own venv; do not run
`wsl.exe`, the PowerShell wrapper, or assume `/opt/nano-vllm` is present.
Read the [operator plan and Linux handoff](SEGMENTED_GQA_PREFILL_PLAN_AND_LINUX_HANDOFF.md)
before engineering work. The WSL table below remains historical reference data,
not proof of this native installation. The pinned Qwen3-4B model was downloaded
outside Git; old WSL HF cache contents and ignored logs/traces have not been
migrated. Preserve the old environment and its artifacts.

### Native setup recorded on 2026-10-06

| Item | Value |
| --- | --- |
| Checkout | `/home/tmz/.codex/worktrees/nanokv-native-linux/nanokv` |
| Branch / base commit | `codex/nanokv-linux-setup` / `e92b08e` |
| OS / kernel | Ubuntu 26.04.1 LTS / `7.0.0-34-generic` |
| Python / uv | Python 3.12.14 / uv 0.12.18 |
| Virtual environment | checkout `.venv` |
| GPU / driver | RTX 3080 Laptop, 16 GiB, SM86 / 595.91.07 open kernel module; CUDA execution verified after power cycling |
| PyTorch / CUDA runtime | 2.7.1+cu128 / 12.8 |
| Triton | 3.3.1 |
| FlashAttention | 2.8.3.post1, official prebuilt `cu12torch2.7`, C++11 ABI true wheel |
| Transformers / xxhash | 4.57.6 / 4.0.1 |
| einops | 0.8.2; FlashAttention's missing declared dependency was installed |
| nano-vLLM | 0.2.0, editable install from this checkout |
| CUDA toolkit | The shared AI venv provides nvcc 13.2.86; no separate 12.8 toolkit is installed in the project venv |
| Model | `/home/tmz/models/Qwen3-4B`; all files match HF snapshot `1cfa9a7208912126459214e8b04321603b3df60c` |
| Download metadata | model-local `.cache/huggingface`; source/hash manifest at model-local `.cache/nanokv/source-manifest.json` |
| Runtime verification | Imports, CUDA allocation/synchronization, FlashAttention forward comparison, Triton JIT compilation/execution, `uv pip check` and a short nano-vLLM Qwen3-4B inference passed |

The CUDA 12.8 runtime matches the historical project reference. FlashAttention
was installed from its prebuilt wheel, so this setup did not compile against
the shared CUDA 13.2 toolkit. The native environment now passes minimal model
execution; a system cost baseline has not been run. Do not treat package imports or
`nvidia-smi` visibility alone as a successful CUDA execution check.

### GPU failure and recovery (2026-10-06)

The user repeated a minimal CUDA allocation from the desktop terminal with
this venv and cleared CUDA environment variables. It failed with
`RuntimeError: No CUDA GPUs are available`. The shared AI venv's PyTorch
2.14.0+cu132 also cannot initialize CUDA; a direct `libcuda.so.1` probe returns
`CUDA_ERROR_NO_DEVICE` (100). This is not evidence of a Codex-only restriction.

Kernel logs show GFW boot failure, `Xid 119` / GSP RPC timeouts, followed by
`Xid 154: GPU Reset Required`. `nvidia-smi -q` reports
`GPU Recovery Action: Reset` and `GPU requires reset` for temperature,
firmware and other queries. Kernel module, firmware and user driver libraries
are all version 595.91.07; the PCI device is bound to NVIDIA and active.
The initial trigger has not been established. The Xid names a WebKit process,
which identifies the active client, not a proven root cause.

Save desktop work, fully shut down and start the laptop again, then rerun the
minimal CUDA check below. NVIDIA's [Xid catalog](https://docs.nvidia.com/deploy/xid-errors/analyzing-xid-catalog.html)
recommends GPU reset or a power cycle for persistent Xid 119/120 errors.
If the error returns after power cycling, collect the new boot's kernel logs
and investigate the driver/GSP issue before changing project packages.
Pre-recovery evidence is saved outside Git at
`/home/tmz/ai/diagnostics/nanokv-gpu-2026-10-06.log`.

After the user power cycled the laptop, temperature/power telemetry returned
to normal. The project venv successfully allocated a CUDA tensor and
synchronized; it reported `2.7.1+cu128`, CUDA 12.8 and SM86. A small causal
FlashAttention forward matched PyTorch SDPA (maximum absolute error 0.0),
and a minimal Triton kernel compiled and produced the expected result.
`uv pip check` passed for all 46 installed packages after installing einops.
No Xid/GSP timeout was found in the current boot's kernel logs at the check.
This verifies minimal execution, not long-running driver stability. Model
inference was subsequently checked as recorded below. The shared 13.2
environment was not separately revalidated after
recovery; use the project venv for NanoKV work.

Activate this checkout with:

```bash
cd /home/tmz/.codex/worktrees/nanokv-native-linux/nanokv
source .venv/bin/activate
unset CUDA_HOME CUDA_PATH CUDACXX LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST=8.6
export NANOVLLM_MODEL=/home/tmz/models/Qwen3-4B
export NANOKV_MODEL="$NANOVLLM_MODEL"
python -c 'import torch; x=torch.ones(1, device="cuda"); torch.cuda.synchronize(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), x.item())'
```

The terminal may start with the shared `~/ai/venv` environment active. Clear
its CUDA 13.2 library paths before using this CUDA 12.8 project venv. Source
`~/ai/env.sh` again when returning to the shared AI environment.

Recent system comparisons use Qwen3-4B snapshot
`1cfa9a7208912126459214e8b04321603b3df60c`; Qwen3-0.6B below remains the
old environment-check model. Pin the same 4B model for comparable system
A/B runs. Operator microbenchmarks do not require a model.

### Model acquisition and execution check (2026-10-06)

The initial Hugging Face download was slow. `hf-mirror.com` redirected the
fixed-revision requests back to Hugging Face during the check. TUNA's published
mirror list did not list a Hugging Face/Qwen model service; its PyPI mirror is
for Python packages. Remaining weights were resumed from
[Qwen's official ModelScope repository](https://modelscope.cn/models/Qwen/Qwen3-4B),
using ModelScope weight revision `8cd0101f70cac4f1efcebc979faf483558e39297`.
All three weight SHA256 values matched the pinned Hugging Face snapshot.
Existing configuration/tokenizer files were retained from that HF snapshot.
Every final file was checked against the official HF SHA256 or Git blob hash;
the complete manifest is stored beside the model, outside Git.

A local/offline nano-vLLM run loaded Qwen3-4B with TP=1, eager execution,
`max_model_len=512`, `max_num_batched_tokens=512`, `max_num_seqs=1`, and
`gpu_memory_utilization=0.7`. A short non-thinking prompt asking `1+1` produced
`2<|im_end|>` and shut down cleanly. The check record is
`/home/tmz/ai/diagnostics/nanokv-qwen3-4b-smoke-2026-10-06.json`.
This is an environment check, not a sparse-attention correctness result,
long-context capacity result or performance benchmark.

## Historical WSL reference (last verified)

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

## Historical Windows and WSL routing note

The older `ext4.vhdx` file is the backing disk for WSL, not a Windows directory containing `/opt`. Windows tools cannot discover `/opt/nano-vllm/.venv/bin/python` by recursively searching drive D. A Windows Python probe therefore says nothing about that WSL environment. This note describes the previous deployment only; native Linux is now the active continuation.

This project depends on the Linux CUDA ecosystem, including Triton, FlashAttention, and NCCL-oriented execution. The previous setup kept Windows as the control plane and WSL as the development and execution plane. That arrangement is not required for the current native Linux checkout.

## Historical WSL invocation

This command applied to the previous Windows coordination repository:

```powershell
.\scripts\wsl-nanovllm.ps1 -Command "python --version"
.\scripts\wsl-nanovllm.ps1 -Command "python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'"
.\scripts\wsl-nanovllm.ps1 -Command "git status --short --branch"
```

The old wrapper selected `NanoVLLM-Ubuntu`, changed to `/opt/nano-vllm`, activated `.venv`, set `CUDA_HOME` and `HF_HOME`, and then ran the supplied command. It is not the canonical invocation for native Linux.

## Historical Windows Codex routing

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
