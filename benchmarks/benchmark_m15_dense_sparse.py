"""Matched dense feature-off versus current sparse runtime benchmark."""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

import psutil
import torch
from transformers import AutoTokenizer

from nanovllm import LLM, SamplingParams


HF_HOME = "/opt/models/.cache/huggingface"
YARN = {"rope_type": "yarn", "type": "yarn", "factor": 4.0,
        "original_max_position_embeddings": 32768, "rope_theta": 1000000}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("dense", "sparse"), required=True)
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--gen-tokens", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault("HF_HOME", HF_HOME)

    model = glob.glob(f"{HF_HOME}/hub/models--Qwen--Qwen3-4B/snapshots/*")[0]
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    unit = tokenizer.encode(
        "A controlled baseline must keep the model prompt length and sampling "
        "configuration identical across both inference paths. "
    )
    prompt = (unit * ((args.seq_len // len(unit)) + 1))[:args.seq_len]
    common = dict(
        model=model, enforce_eager=True, tensor_parallel_size=1, max_num_seqs=1,
        max_num_batched_tokens=args.seq_len,
        max_model_len=args.seq_len + args.gen_tokens + 16, dtype="bfloat16",
        rope_scaling_override=YARN,
    )
    if args.mode == "sparse":
        common.update(
            enable_sparse_attention=True, use_m12_runtime=True,
            sparse_selector="query_guided", sparse_retrieval_block_size=64,
            sparse_num_representatives=4, sparse_recent_tokens=512,
            sparse_first_tokens=64, sparse_top_k=32,
            sparse_prefill_chunk_size=4096, sparse_prefill_query_segments=1,
            sparse_prefill_attention_backend="flash",
            sparse_gather_index_select=True,
        )

    llm = LLM(**common)
    process = psutil.Process()
    rss_before = process.memory_info().rss
    torch.cuda.reset_peak_memory_stats()
    try:
        llm.add_request(prompt, SamplingParams(
            max_tokens=args.gen_tokens, temperature=0.0, ignore_eos=True))
        prefill_ms: list[float] = []
        decode_ms: list[float] = []
        output = []
        while not llm.is_finished():
            torch.cuda.synchronize()
            start = time.perf_counter()
            output, num_scheduled_tokens = llm.step()
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000.0
            (prefill_ms if num_scheduled_tokens > 0 else decode_ms).append(elapsed)
        steady = decode_ms[4:] if len(decode_ms) > 4 else decode_ms
        token_ids = list(output[0][1]) if output else []
        result = {
            "measurement": "CUDA-synchronized engine steps with prefill separated",
            "config": {"mode": args.mode, "model": model,
                       "seq_len": args.seq_len, "gen_tokens": args.gen_tokens,
                       "temperature": 0.0, "yarn_factor": 4.0},
            "prefill": {"steps": len(prefill_ms), "step_ms": prefill_ms,
                        "wall_ms": sum(prefill_ms)},
            "decode": {
                "steps": len(decode_ms), "step_ms": decode_ms,
                "mean_ms": statistics.mean(decode_ms),
                "median_ms": statistics.median(decode_ms),
                "steady_drop4_mean_ms": statistics.mean(steady),
                "steady_drop4_median_ms": statistics.median(steady),
            },
            "generated_token_ids": token_ids,
            "generated_text": tokenizer.decode(token_ids),
            "runtime": {
                "host_rss_before_gib": rss_before / 1024**3,
                "host_rss_after_gib": process.memory_info().rss / 1024**3,
                "host_available_after_gib": psutil.virtual_memory().available / 1024**3,
                "gpu_allocated_gib": torch.cuda.memory_allocated() / 1024**3,
                "gpu_reserved_gib": torch.cuda.memory_reserved() / 1024**3,
                "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
            },
            "git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True).strip(),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(args.output)
        print(json.dumps(result, indent=2))
    finally:
        llm.exit()


if __name__ == "__main__":
    main()
