"""Validate Milestone 1 metrics against real GPU prefix-cache behavior."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from nanovllm import LLM, SamplingParams


def make_prompt(first: list[int] | None = None, change_at: int | None = None) -> list[int]:
    tokens = [1000 + (index % 97) for index in range(600)]
    if first is not None:
        tokens[: len(first)] = first
    if change_at is not None:
        for index in range(change_at, len(tokens)):
            tokens[index] = 3000 + (index % 89)
    return tokens


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--enable-reusable-cache", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/instrumentation_validation.json"),
    )
    args = parser.parse_args()

    sampling = SamplingParams(temperature=0.0, max_tokens=2, ignore_eos=True)
    llm = LLM(
        str(args.model),
        enforce_eager=True,
        tensor_parallel_size=1,
        enable_cache_metrics=True,
        max_model_len=1024,
        enable_reusable_cache=args.enable_reusable_cache,
    )

    base = make_prompt()
    cases = [
        ("cold_miss", base, 0, 600, 0),
        ("same_prompt_gpu_hit", base, 512, 88, 2),
        ("one_block_partial_hit", make_prompt(base[:256], 256), 256, 344, 1),
        ("non_aligned_300_token_prefix", make_prompt(base[:300], 300), 256, 344, 1),
        ("completely_different_prompt", [5000 + (i % 83) for i in range(600)], 0, 600, 0),
    ]

    results = []
    cold_token_ids = None
    for name, prompt, reused_tokens, recomputed_tokens, gpu_hit_blocks in cases:
        output = llm.generate([prompt], sampling, use_tqdm=False)[0]
        metrics = llm.get_cache_metrics()
        assert metrics["requests"] == 1, (name, metrics)
        assert metrics["reused_tokens"] == reused_tokens, (name, metrics)
        assert metrics["recomputed_tokens"] == recomputed_tokens, (name, metrics)
        assert metrics["gpu_hit_blocks"] == gpu_hit_blocks, (name, metrics)
        assert metrics["lookup_time_ms"] >= 0, (name, metrics)
        assert metrics["prefill_time_ms"] > 0, (name, metrics)
        assert metrics["decode_time_ms"] > 0, (name, metrics)
        assert metrics["ttft_ms"] > 0, (name, metrics)
        if name == "cold_miss":
            cold_token_ids = output["token_ids"]
        elif name == "same_prompt_gpu_hit":
            assert output["token_ids"] == cold_token_ids
        results.append(
            {
                "name": name,
                "prompt_tokens": len(prompt),
                "output_token_ids": output["token_ids"],
                "metrics": metrics,
            }
        )

    result = {
        "model": str(args.model),
        "device": torch.cuda.get_device_name(),
        "block_size": llm.model_runner.config.kvcache_block_size,
        "reusable_cache_enabled": args.enable_reusable_cache,
        "sampling": {"temperature": 0.0, "max_tokens": 2, "ignore_eos": True},
        "cases": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
