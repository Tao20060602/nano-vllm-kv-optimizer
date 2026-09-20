#!/usr/bin/env bash
set -euo pipefail

expected_repo=/opt/nano-vllm
expected_venv=/opt/nano-vllm/.venv
expected_model=/opt/models/Qwen3-0.6B

cd "$expected_repo"

test -x "$expected_venv/bin/python"
test -d "$expected_model"
test "${VIRTUAL_ENV:-}" = "$expected_venv"

echo "distribution=NanoVLLM-Ubuntu"
echo "repo=$(pwd)"
echo "git_head=$(git rev-parse --short HEAD)"
echo "git_status=$(git status --short --branch | head -n 1)"
echo "python=$($expected_venv/bin/python --version 2>&1)"
echo "uv=$(command -v uv)"
echo "cuda_home=${CUDA_HOME:-unset}"
echo "model=$expected_model"
echo "hf_home=${HF_HOME:-unset}"

$expected_venv/bin/python - <<'PY'
import flash_attn
import torch
import triton

print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"cuda_runtime={torch.version.cuda}")
print(f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
print(f"compute_capability={'.'.join(map(str, torch.cuda.get_device_capability(0))) if torch.cuda.is_available() else 'none'}")
print(f"triton={triton.__version__}")
print(f"flash_attn={flash_attn.__version__}")
PY
