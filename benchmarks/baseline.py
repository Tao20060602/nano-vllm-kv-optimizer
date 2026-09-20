"""Record reproducible environment and baseline prerequisites.

This script deliberately does not invent latency or throughput if the runtime,
model, or GPU is unavailable. It is safe to run before installing dependencies.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/baseline_environment.json"))
    parser.add_argument("--model", type=Path, default=Path.home() / "huggingface" / "Qwen3-0.6B")
    args = parser.parse_args()

    packages = {name: importlib.util.find_spec(name) is not None for name in ("torch", "transformers", "triton", "flash_attn", "xxhash")}
    torch_info: dict[str, object] = {"installed": packages["torch"]}
    if packages["torch"]:
        import torch

        torch_info.update(
            {
                "version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda,
                "device_count": torch.cuda.device_count(),
                "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            }
        )
    result = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": sys.version,
        "executable": sys.executable,
        "packages": packages,
        "torch": torch_info,
        "model_path": str(args.model),
        "model_exists": args.model.is_dir(),
        "git_head": command_output(["git", "rev-parse", "HEAD"]),
        "git_status": command_output(["git", "status", "--short", "--branch"]),
        "baseline_runs": {
            "example.py": "not_run_by_probe",
            "bench.py": "not_run_by_probe",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
