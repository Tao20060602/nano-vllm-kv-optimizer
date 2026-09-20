param(
    [string]$Command = "exec bash"
)

$ErrorActionPreference = "Stop"
$distribution = "NanoVLLM-Ubuntu"
$repository = "/opt/nano-vllm"
$environment = "/opt/nano-vllm/.venv"
$modelCache = "/opt/models/.cache/huggingface"
$cudaHome = "/usr/local/cuda-12.8"

$bootstrap = @"
set -e
cd '$repository'
source '$environment/bin/activate'
export HF_HOME='$modelCache'
export CUDA_HOME='$cudaHome'
$Command
"@

& wsl.exe -d $distribution -- bash -lc $bootstrap
exit $LASTEXITCODE
