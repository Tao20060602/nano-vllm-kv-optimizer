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
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(os.environ.get("NANOVLLM_MODEL", Path.home() / "huggingface" / "Qwen3-0.6B")),
    )
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/deterministic_baseline.json"))
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--enable-cache-metrics", action="store_true")
    args = parser.parse_args()
    if not args.model.is_dir():
        raise SystemExit(f"model directory does not exist: {args.model}")

    import torch
    from nanovllm import LLM, SamplingParams

    # Token IDs are deliberately explicit so tokenizer chat-template changes do
    # not silently change the baseline input.
    prompt_token_ids = [151644, 872, 198, 151645, 198, 151644, 77091, 198]
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, ignore_eos=True)
    llm = LLM(
        str(args.model),
        enforce_eager=True,
        tensor_parallel_size=1,
        enable_cache_metrics=args.enable_cache_metrics,
    )
    torch.cuda.synchronize()
    start = time.perf_counter()
    outputs = llm.generate([prompt_token_ids], sampling, use_tqdm=False)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    config = llm.model_runner.config
    kv_cache = llm.model_runner.kv_cache
    hf_config = config.hf_config
    block_bytes = (
        2
        * hf_config.num_hidden_layers
        * config.kvcache_block_size
        * (hf_config.num_key_value_heads // config.tensor_parallel_size)
        * hf_config.head_dim
        * hf_config.dtype.itemsize
    )
    result = {
        "model": str(args.model),
        "prompt_token_ids": prompt_token_ids,
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "cache_metrics_enabled": args.enable_cache_metrics,
        "output_token_ids": outputs[0]["token_ids"],
        "elapsed_seconds": elapsed,
        "cuda_device": torch.cuda.get_device_name(),
        "kv_cache": {
            "shape": list(kv_cache.shape),
            "dtype": str(kv_cache.dtype),
            "device": str(kv_cache.device),
            "block_size_tokens": config.kvcache_block_size,
            "num_gpu_blocks": config.num_kvcache_blocks,
            "bytes_per_complete_block": block_bytes,
        },
    }
    if args.enable_cache_metrics:
        result["cache_metrics"] = llm.get_cache_metrics()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
