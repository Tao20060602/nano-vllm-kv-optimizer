"""Reproducible raw benchmark for cold, GPU-hit, and CPU-hit prefix reuse."""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoConfig

from nanovllm import LLM, SamplingParams
from nanovllm.kvdb.fingerprint import build_cache_fingerprint


BREAK_EVEN_PREFIXES = (256, 512, 1024, 2048, 4096)


def kv_block_bytes(model: Path) -> int:
    config = AutoConfig.from_pretrained(model)
    fingerprint = build_cache_fingerprint(
        str(model), config, block_size=256, tensor_parallel_size=1
    )
    dtype = getattr(torch, fingerprint.dtype.removeprefix("torch."))
    return (
        2
        * fingerprint.num_layers
        * fingerprint.block_size
        * fingerprint.num_kv_heads
        * fingerprint.head_dim
        * dtype.itemsize
    )


def exact_prompt(prefix_tokens: int, suffix_tokens: int, namespace: int) -> list[int]:
    prefix = [1000 + namespace * 100 + (i % 97) for i in range(prefix_tokens)]
    suffix = [20000 + namespace * 100 + (i % 89) for i in range(suffix_tokens)]
    return prefix + suffix


def partial_prompt(
    total_tokens: int,
    shared_tokens: int,
    namespace: int,
    variant: int,
) -> list[int]:
    shared = [30000 + namespace * 500 + (i % 97) for i in range(shared_tokens)]
    suffix = [
        50000 + namespace * 1000 + variant * 113 + (i % 101)
        for i in range(total_tokens - shared_tokens)
    ]
    return shared + suffix


def measure(
    llm: LLM,
    prompt: list[int],
    sampling: SamplingParams,
    *,
    mode: str,
    point: dict,
    iteration: int,
    warmup: bool,
) -> dict:
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    output = llm.generate([prompt], sampling, use_tqdm=False)[0]
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - start) * 1000.0
    return {
        "mode": mode,
        **point,
        "iteration": iteration,
        "warmup": warmup,
        "wall_time_ms": wall_ms,
        "output_token_ids": output["token_ids"],
        "metrics": llm.get_cache_metrics(),
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "cpu_cache": llm.get_cpu_cache_stats(),
    }


def run_exact_mode(
    llm: LLM,
    mode: str,
    warmups: int,
    repetitions: int,
) -> list[dict]:
    records = []
    points = [
        (prefix, 16, 1) for prefix in BREAK_EVEN_PREFIXES
    ] + [(1024, 64, 32)]
    for point_id, (prefix_tokens, suffix_tokens, max_tokens) in enumerate(points):
        prompt = exact_prompt(prefix_tokens, suffix_tokens, point_id)
        sampling = SamplingParams(
            temperature=0.0, max_tokens=max_tokens, ignore_eos=True
        )
        point = {
            "point": "break_even" if max_tokens == 1 else "full_request_32",
            "prefix_tokens": prefix_tokens,
            "suffix_tokens": suffix_tokens,
            "prompt_tokens": len(prompt),
            "reuse_ratio": prefix_tokens / len(prompt),
            "max_tokens": max_tokens,
        }
        if mode != "cold":
            llm.generate([prompt], sampling, use_tqdm=False)
            llm.clear_gpu_prefix_cache() if mode.startswith("cpu_") else None
        total = warmups + repetitions
        for iteration in range(total):
            if mode == "cold":
                llm.clear_gpu_prefix_cache()
            elif mode.startswith("cpu_"):
                llm.clear_gpu_prefix_cache()
            record = measure(
                llm,
                prompt,
                sampling,
                mode=mode,
                point=point,
                iteration=iteration,
                warmup=iteration < warmups,
            )
            expected_reused = prefix_tokens
            if mode == "cold":
                expected_reused = 0
            assert record["metrics"]["reused_tokens"] == expected_reused
            assert record["metrics"]["prefill_executed_tokens"] == (
                len(prompt) - expected_reused
            )
            records.append(record)
        print(f"completed {mode}: prefix={prefix_tokens}, suffix={suffix_tokens}, output={max_tokens}", flush=True)
    return records


def run_partial_mode(
    llm: LLM,
    warmups: int,
    repetitions: int,
) -> list[dict]:
    records = []
    total_tokens = 2064
    full_blocks = 8
    for ratio_index, ratio in enumerate((0.25, 0.5, 0.75, 1.0)):
        shared_blocks = math.floor(full_blocks * ratio)
        shared_tokens = shared_blocks * 256
        namespace = 20 + ratio_index
        prime = partial_prompt(total_tokens, shared_tokens, namespace, 0)
        sampling = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)
        llm.generate([prime], sampling, use_tqdm=False)
        llm.clear_gpu_prefix_cache()
        total = warmups + repetitions
        for iteration in range(total):
            variant = iteration + 1 if ratio < 1.0 else 0
            prompt = partial_prompt(total_tokens, shared_tokens, namespace, variant)
            point = {
                "point": "partial_reuse",
                "prefix_tokens": shared_tokens,
                "suffix_tokens": total_tokens - shared_tokens,
                "prompt_tokens": total_tokens,
                "reuse_ratio": ratio,
                "max_tokens": 1,
            }
            record = measure(
                llm,
                prompt,
                sampling,
                mode="cpu_partial_pinned",
                point=point,
                iteration=iteration,
                warmup=iteration < warmups,
            )
            assert record["metrics"]["cpu_hit_blocks"] == shared_blocks
            assert record["metrics"]["prefill_executed_tokens"] == (
                total_tokens - shared_tokens
            )
            records.append(record)
            llm.clear_gpu_prefix_cache()
        print(f"completed partial pinned: reuse_ratio={ratio:.2f}", flush=True)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("cold", "gpu", "cpu_pageable", "cpu_pinned", "cpu_partial_pinned"),
        required=True,
    )
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.warmups < 5 or args.repetitions < 20:
        raise SystemExit("benchmark requires at least 5 warmups and 20 measurements")

    cpu_mode = args.mode.startswith("cpu_")
    block_bytes = kv_block_bytes(args.model)
    llm = LLM(
        str(args.model),
        enforce_eager=True,
        tensor_parallel_size=1,
        enable_cache_metrics=True,
        enable_reusable_cache=args.mode == "gpu",
        enable_cpu_cache=cpu_mode,
        cpu_cache_capacity_bytes=64 * block_bytes,
        cpu_cache_pinned=args.mode in ("cpu_pinned", "cpu_partial_pinned"),
        max_model_len=4352,
    )
    if args.mode == "cpu_partial_pinned":
        records = run_partial_mode(llm, args.warmups, args.repetitions)
    else:
        records = run_exact_mode(llm, args.mode, args.warmups, args.repetitions)

    result = {
        "schema_version": 1,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "model": str(args.model),
        "python_platform": platform.platform(),
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "warmups": args.warmups,
        "repetitions": args.repetitions,
        "block_size": 256,
        "bytes_per_block": block_bytes,
        "not_run_prefix_tokens": [8192],
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(records)} raw records to {args.output}")


if __name__ == "__main__":
    main()
