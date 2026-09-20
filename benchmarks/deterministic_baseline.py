"""Deterministic greedy baseline for later NanoKV comparisons.

The script uses explicit token IDs, a fixed output length, and temperature zero.
It records token IDs and timings as JSON. It intentionally fails with a precise
message when the model/runtime is unavailable instead of fabricating a result.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path.home() / "huggingface" / "Qwen3-0.6B")
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/deterministic_baseline.json"))
    parser.add_argument("--max-tokens", type=int, default=1)
    args = parser.parse_args()
    if not args.model.is_dir():
        raise SystemExit(f"model directory does not exist: {args.model}")

    import torch
    from nanovllm import LLM, SamplingParams

    # Token IDs are deliberately explicit so tokenizer chat-template changes do
    # not silently change the baseline input.
    prompt_token_ids = [151644, 872, 198, 151645, 198, 151644, 77091, 198]
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, ignore_eos=True)
    llm = LLM(str(args.model), enforce_eager=True, tensor_parallel_size=1)
    torch.cuda.synchronize()
    start = time.perf_counter()
    outputs = llm.generate([prompt_token_ids], sampling, use_tqdm=False)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    result = {
        "model": str(args.model),
        "prompt_token_ids": prompt_token_ids,
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "output_token_ids": outputs[0]["token_ids"],
        "elapsed_seconds": elapsed,
        "cuda_device": torch.cuda.get_device_name(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
