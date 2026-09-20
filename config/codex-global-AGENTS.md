# Global project routing

## NanoKV / nano-vLLM / AlayaDB work

When a task concerns NanoKV, nano-vLLM, the AlayaDB internship project, or the Windows workspace `C:\Users\28898\Documents\ChatGPT\nano-vllm找实习`:

- Use WSL distribution `NanoVLLM-Ubuntu` explicitly; never substitute the generic `Ubuntu` distribution.
- Treat `/opt/nano-vllm` as the authoritative Git worktree and `/opt/nano-vllm/.venv` as the authoritative Python environment.
- Use `/opt/models/Qwen3-0.6B` and `HF_HOME=/opt/models/.cache/huggingface`.
- Run commands through `wsl.exe -d NanoVLLM-Ubuntu -- bash -lc ...` or the repository wrapper `scripts/wsl-nanovllm.ps1`.
- Do not diagnose this project using Windows Python and do not install duplicate native-Windows CUDA/PyTorch/Triton/FlashAttention/model copies.
- Read the repository `AGENTS.md` and `docs/environment.md` before environment changes.

These rules are project-specific and must not redirect unrelated repositories into NanoVLLM-Ubuntu.
