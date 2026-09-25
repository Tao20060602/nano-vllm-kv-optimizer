"""Matched sparse-decode benchmark with synchronized step boundaries."""

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


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * p)
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=32768)
    parser.add_argument("--gen-tokens", type=int, default=32)
    parser.add_argument("--index-select", type=int, choices=(0, 1), required=True)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--dynamic-top-k", action="store_true")
    parser.add_argument("--dynamic-mass", type=float, default=0.90)
    parser.add_argument(
        "--check-finite-output", action="store_true",
        help="enable the per-layer finite check (diagnostic baseline; synchronizes CUDA)",
    )
    parser.add_argument(
        "--cuda-stage-profile", action="store_true",
        help="record CUDA events for the final decode step; adds profiling overhead",
    )
    parser.add_argument(
        "--trace-output", type=Path,
        help="optional PyTorch CPU/CUDA trace for one steady decode step",
    )
    parser.add_argument(
        "--trace-step", type=int, default=4,
        help="zero-based decode step to capture after the first four steps",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.trace_output is not None
            and (args.trace_step < 0 or args.trace_step >= args.gen_tokens - 1)):
        raise SystemExit("--trace-step must identify a generated decode step")
    os.environ.setdefault("HF_HOME", HF_HOME)

    model = glob.glob(f"{HF_HOME}/hub/models--Qwen--Qwen3-4B/snapshots/*")[0]
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    unit = tokenizer.encode(
        "The careful engineer measured memory bandwidth and attention sparsity "
        "before shipping the optimized inference kernel. "
    )
    prompt = (unit * ((args.seq_len // len(unit)) + 1))[:args.seq_len]
    llm = LLM(
        model, enforce_eager=True, tensor_parallel_size=1, max_num_seqs=1,
        max_model_len=131072, dtype="bfloat16", enable_sparse_attention=True,
        use_m12_runtime=True, sparse_selector="query_guided",
        sparse_retrieval_block_size=64, sparse_num_representatives=4,
        sparse_recent_tokens=512, sparse_first_tokens=64, sparse_top_k=32,
        sparse_decode_top_k=args.top_k,
        sparse_dynamic_top_k=args.dynamic_top_k,
        sparse_dynamic_top_k_mass=args.dynamic_mass,
        sparse_prefill_chunk_size=4096, sparse_prefill_query_segments=1,
        sparse_prefill_attention_backend="flash",
        sparse_gather_index_select=bool(args.index_select),
        sparse_check_finite_outputs=args.check_finite_output,
        rope_scaling_override=YARN,
    )
    process = psutil.Process()
    rss_before = process.memory_info().rss
    torch.cuda.reset_peak_memory_stats()
    try:
        layers = [m.sparse_rt for m in llm.model_runner.model.modules()
                  if getattr(m, "sparse_rt", None) is not None]
        expected_prefill_steps = (args.seq_len + 4095) // 4096
        llm.add_request(prompt, SamplingParams(
            max_tokens=args.gen_tokens, temperature=0.0, ignore_eos=True))
        prefill_ms: list[float] = []
        decode_ms: list[float] = []
        decode_k_blocks_by_step: list[list[int]] = []
        decode_h2d_mib_by_step: list[float] = []
        output = []
        profiled_decode_step = None
        profile_summary = None
        while not llm.is_finished():
            if args.cuda_stage_profile and len(decode_ms) == args.gen_tokens - 2:
                for layer in layers:
                    layer.profile_cuda_stages = True
            should_profile = (
                args.trace_output is not None
                and profiled_decode_step is None
                and len(prefill_ms) == expected_prefill_steps
                and len(decode_ms) == args.trace_step
            )
            if should_profile:
                for layer in layers:
                    layer.profile_torch_stages = True
                torch.cuda.synchronize()
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    record_shapes=False,
                    profile_memory=False,
                    with_stack=False,
                ) as profiler:
                    start = time.perf_counter()
                    output, num_scheduled_tokens = llm.step()
                    torch.cuda.synchronize()
                    elapsed = (time.perf_counter() - start) * 1000.0
                for layer in layers:
                    layer.profile_torch_stages = False
                if num_scheduled_tokens >= 0:
                    raise RuntimeError(
                        "trace point was expected to be a decode step, but scheduler "
                        "reported prefill work"
                    )
                profiled_decode_step = len(decode_ms)
                args.trace_output.parent.mkdir(parents=True, exist_ok=True)
                profiler.export_chrome_trace(str(args.trace_output))
                profile_summary = profiler.key_averages().table(
                    sort_by="self_cuda_time_total", row_limit=40
                )
                args.trace_output.with_suffix(
                    args.trace_output.suffix + ".summary.txt"
                ).write_text(profile_summary + "\n", encoding="utf-8")
            else:
                torch.cuda.synchronize()
                start = time.perf_counter()
                output, num_scheduled_tokens = llm.step()
                torch.cuda.synchronize()
                elapsed = (time.perf_counter() - start) * 1000.0
            (prefill_ms if num_scheduled_tokens > 0 else decode_ms).append(elapsed)
            if num_scheduled_tokens < 0:
                selected_tokens_this_step = [
                    layer.timings["selected_tokens"] for layer in layers
                ]
                decode_k_blocks_by_step.append([
                    tokens // 64 for tokens in selected_tokens_this_step
                ])
                decode_h2d_mib_by_step.append(
                    sum(selected_tokens_this_step) * 2 * 8 * 128 * 2 / 1024**2
                )

        if args.trace_output is not None and profiled_decode_step is None:
            raise RuntimeError("requested decode trace point was not reached")

        steady = [
            elapsed for index, elapsed in enumerate(decode_ms)
            if index >= 4 and index != profiled_decode_step
        ] or decode_ms
        stage_names = ("selector_ms", "cpu_gather_ms", "h2d_pack_ms",
                       "recent_ms", "packed_attn_ms")
        last_stage = {name: sum(layer.timings.get(name, 0.0) for layer in layers)
                      for name in stage_names}
        cuda_stage = ({
            name: sum(layer.cuda_stage_events[name][0].elapsed_time(
                layer.cuda_stage_events[name][1]) for layer in layers)
            for name in ("selector", "recent", "h2d_pack", "packed_attention")
        } if args.cuda_stage_profile else {})
        result = {
            "measurement": "CUDA-synchronized engine steps; decode excludes prefill",
            "config": {
                "model": model, "seq_len": args.seq_len,
                "gen_tokens": args.gen_tokens,
                "index_select": bool(args.index_select), "chunk_size": 4096,
                "check_finite_output": args.check_finite_output,
                "query_segments": 1, "prefill_backend": "flash",
                "block_size": 64, "representatives": 4,
                "prefill_top_k_blocks": 32, "decode_top_k_blocks": args.top_k,
                "dynamic_top_k": args.dynamic_top_k,
                "dynamic_mass": args.dynamic_mass,
                "sink_tokens": 64, "recent_tokens": 512,
            },
            "prefill": {"steps": len(prefill_ms), "wall_ms": sum(prefill_ms)},
            "decode": {
                "steps": len(decode_ms), "step_ms": decode_ms,
                "mean_ms": statistics.mean(decode_ms),
                "median_ms": statistics.median(decode_ms),
                "p95_ms": percentile(decode_ms, 0.95),
                "steady_drop4_mean_ms": statistics.mean(steady),
                "steady_drop4_median_ms": statistics.median(steady),
            },
            "last_decode_stage_ms_36_layers": last_stage,
            "last_decode_cuda_event_ms_36_layers": cuda_stage,
            "profiling": {
                "trace_output": str(args.trace_output) if args.trace_output else None,
                "trace_step_index": profiled_decode_step,
                "profiled_step_excluded_from_steady_summary": True,
                "trace_is_diagnostic_not_a_performance_run": True,
            },
            "payload": {
                "selected_tokens_per_layer": [
                    layer.timings["selected_tokens"] for layer in layers
                ],
                "selected_blocks_per_layer": [
                    layer.timings["selected_tokens"] // 64 for layer in layers
                ],
                "selected_blocks_by_decode_step": decode_k_blocks_by_step,
                "selected_blocks_histogram": {
                    str(k): sum(row.count(k) for row in decode_k_blocks_by_step)
                    for k in sorted({k for row in decode_k_blocks_by_step for k in row})
                },
                "selected_blocks_mean": statistics.mean(
                    k for row in decode_k_blocks_by_step for k in row
                ) if decode_k_blocks_by_step else None,
                "selected_h2d_mib_per_decode_token": (
                    sum(layer.timings["selected_tokens"] for layer in layers)
                    * 2 * 8 * 128 * 2 / 1024**2),
                "selected_h2d_mib_by_decode_step": decode_h2d_mib_by_step,
            },
            "generated_token_ids": list(output[0][1]) if output else [],
            "runtime": {
                "layers": len(layers),
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
