"""Matched sparse-decode benchmark with synchronized step boundaries."""

from __future__ import annotations

import argparse
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
    parser.add_argument(
        "--cpu-threads", type=int,
        help="explicit PyTorch intra-op thread count; absent keeps the process default",
    )
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--kv-pipeline", action="store_true",
                        help="enqueue K H2D before CPU V gather during decode")
    parser.add_argument("--gather-abba", action="store_true",
                        help="warm four pipeline steps, then serial/pipeline/pipeline/serial")
    parser.add_argument("--dynamic-top-k", action="store_true")
    parser.add_argument("--dynamic-mass", type=float, default=0.90)
    parser.add_argument(
        "--model",
        default=(f"{HF_HOME}/hub/models--Qwen--Qwen3-4B/snapshots/"
                 "1cfa9a7208912126459214e8b04321603b3df60c"),
        help="model path or Hugging Face model ID",
    )
    parser.add_argument(
        "--prompt-file", type=Path,
        help="UTF-8 prompt file; tokenized as-is and must contain at least --seq-len tokens",
    )
    parser.add_argument(
        "--selector-cuda-graph", action="store_true",
        help="capture sparse selector calls with CUDA graphs",
    )
    parser.add_argument(
        "--selector-static-mask", action="store_true",
        help="cache valid protected-block indices for eager decode selection",
    )
    parser.add_argument(
        "--fused-selector", action="store_true",
        help="use the fused Triton block-scoring kernel for the flat selector",
    )
    parser.add_argument(
        "--quant-history", action="store_true",
        help="store the CPU history as int8 (K per-channel, V per-block)",
    )
    parser.add_argument(
        "--gather-sort", action="store_true",
        help="sort selected block ids before the CPU gather",
    )
    parser.add_argument(
        "--fused-dequant", action="store_true",
        help="fuse int8 dequantize into one kernel (requires --quant-history)",
    )
    parser.add_argument(
        "--selector-abba", action="store_true",
        help=("within one run, warm four graph decode steps then alternate "
              "eager/graph/graph/eager for steady decode timing"),
    )
    parser.add_argument(
        "--record-selected-ids", action="store_true",
        help="record selected block IDs per layer and decode step for exact comparisons",
    )
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
    parser.add_argument(
        "--nsys-trace-step", type=int,
        help="zero-based decode step for Nsight Systems cudaProfilerApi capture",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.selector_static_mask and args.selector_cuda_graph:
        raise SystemExit("choose one selector optimization per comparison")
    if args.kv_pipeline and not args.index_select:
        raise SystemExit("--kv-pipeline requires --index-select 1")
    if args.gather_abba:
        if not args.kv_pipeline:
            raise SystemExit("--gather-abba requires --kv-pipeline")
        if (args.selector_abba or args.cuda_stage_profile or args.trace_output
                or args.nsys_trace_step is not None or args.check_finite_output):
            raise SystemExit("--gather-abba cannot combine with other ABBA/profiling/finite checks")
    if args.cpu_threads is not None:
        if args.cpu_threads < 1:
            raise SystemExit("--cpu-threads must be positive")
        torch.set_num_threads(args.cpu_threads)
    if (args.trace_output is not None
            and (args.trace_step < 0 or args.trace_step >= args.gen_tokens - 1)):
        raise SystemExit("--trace-step must identify a generated decode step")
    if (args.nsys_trace_step is not None
            and (args.nsys_trace_step < 0
                 or args.nsys_trace_step >= args.gen_tokens - 1)):
        raise SystemExit("--nsys-trace-step must identify a generated decode step")
    if args.nsys_trace_step is not None and args.trace_output is not None:
        raise SystemExit("choose one profiling backend per run")
    if args.selector_abba and not args.selector_cuda_graph:
        raise SystemExit("--selector-abba requires --selector-cuda-graph")
    if args.selector_abba and (
            args.cuda_stage_profile or args.trace_output is not None
            or args.nsys_trace_step is not None or args.check_finite_output):
        raise SystemExit(
            "--selector-abba cannot be combined with profiler or finite-check options"
        )
    os.environ.setdefault("HF_HOME", HF_HOME)

    model = args.model
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    if args.prompt_file is None:
        unit = tokenizer.encode(
            "The careful engineer measured memory bandwidth and attention sparsity "
            "before shipping the optimized inference kernel. "
        )
        prompt = (unit * ((args.seq_len // len(unit)) + 1))[:args.seq_len]
    else:
        prompt_text = args.prompt_file.read_text(encoding="utf-8")
        prompt = tokenizer.encode(prompt_text)
        if len(prompt) < args.seq_len:
            raise SystemExit(
                f"prompt file {args.prompt_file} tokenizes to {len(prompt)} tokens; "
                f"--seq-len={args.seq_len} requires at least that many"
            )
        prompt = prompt[:args.seq_len]
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
        sparse_decode_kv_pipeline=args.kv_pipeline,
        sparse_selector_cuda_graph=args.selector_cuda_graph,
        sparse_selector_static_mask=args.selector_static_mask,
        sparse_fused_selector=args.fused_selector,
        sparse_quant_history=args.quant_history,
        sparse_gather_sort=args.gather_sort,
        sparse_fused_dequant=args.fused_dequant,
        sparse_check_finite_outputs=args.check_finite_output,
        rope_scaling_override=YARN,
    )
    process = psutil.Process()
    rss_before = process.memory_info().rss
    torch.cuda.reset_peak_memory_stats()
    try:
        layers = [m.sparse_rt for m in llm.model_runner.model.modules()
                  if getattr(m, "sparse_rt", None) is not None]
        if args.record_selected_ids:
            for layer in layers:
                layer.record_ids = True
        expected_prefill_steps = (args.seq_len + 4095) // 4096
        llm.add_request(prompt, SamplingParams(
            max_tokens=args.gen_tokens, temperature=0.0, ignore_eos=True))
        prefill_ms: list[float] = []
        decode_ms: list[float] = []
        decode_k_blocks_by_step: list[list[int]] = []
        decode_h2d_mib_by_step: list[float] = []
        selector_abba_enabled_by_step: list[bool] = []
        selector_abba_selector_ms_by_step: list[float] = []
        gather_abba_enabled_by_step: list[bool] = []
        gather_abba_ms_by_step: list[float] = []
        output = []
        profiled_decode_step = None
        nsys_profiled_decode_step = None
        profile_summary = None
        while not llm.is_finished():
            abba_pipeline_enabled = None
            if args.gather_abba and len(prefill_ms) == expected_prefill_steps:
                decode_index = len(decode_ms)
                abba_pipeline_enabled = (decode_index < 4 or (decode_index - 4) % 4 in (1, 2))
                for layer in layers:
                    layer.cfg.decode_kv_pipeline = abba_pipeline_enabled
            abba_graph_enabled = None
            if args.selector_abba and len(prefill_ms) == expected_prefill_steps:
                decode_index = len(decode_ms)
                if decode_index < 4:
                    abba_graph_enabled = True
                else:
                    abba_graph_enabled = (decode_index - 4) % 4 in (1, 2)
                for layer in layers:
                    layer.cfg.selector_cuda_graph = abba_graph_enabled
            if args.cuda_stage_profile and len(decode_ms) == args.gen_tokens - 2:
                for layer in layers:
                    layer.profile_cuda_stages = True
            should_profile = (
                args.trace_output is not None
                and profiled_decode_step is None
                and len(prefill_ms) == expected_prefill_steps
                and len(decode_ms) == args.trace_step
            )
            should_nsys_profile = (
                args.nsys_trace_step is not None
                and nsys_profiled_decode_step is None
                and len(prefill_ms) == expected_prefill_steps
                and len(decode_ms) == args.nsys_trace_step
            )
            if should_nsys_profile:
                for layer in layers:
                    layer.profile_nsys_stages = True
                torch.cuda.synchronize()
                torch.cuda.profiler.start()
                try:
                    with torch.cuda.nvtx.range("nanokv.decode.step"):
                        start = time.perf_counter()
                        output, num_scheduled_tokens = llm.step()
                        torch.cuda.synchronize()
                        elapsed = (time.perf_counter() - start) * 1000.0
                finally:
                    torch.cuda.profiler.stop()
                    for layer in layers:
                        layer.profile_nsys_stages = False
                if num_scheduled_tokens >= 0:
                    raise RuntimeError(
                        "Nsight trace point was expected to be a decode step"
                    )
                nsys_profiled_decode_step = len(decode_ms)
            elif should_profile:
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
                if args.gather_abba:
                    gather_abba_enabled_by_step.append(bool(abba_pipeline_enabled))
                    gather_abba_ms_by_step.append(sum(
                        layer.timings["cpu_gather_ms"] for layer in layers))
                if args.selector_abba:
                    selector_abba_enabled_by_step.append(bool(abba_graph_enabled))
                    selector_abba_selector_ms_by_step.append(sum(
                        layer.timings["selector_ms"] for layer in layers
                    ))
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
        if args.nsys_trace_step is not None and nsys_profiled_decode_step is None:
            raise RuntimeError("requested Nsight decode step was not reached")

        steady = [
            elapsed for index, elapsed in enumerate(decode_ms)
            if index >= 4
            and index != profiled_decode_step
            and index != nsys_profiled_decode_step
        ] or decode_ms
        graph_build_step_index = 0 if args.selector_cuda_graph and decode_ms else None
        steady_reporting = [
            elapsed for index, elapsed in enumerate(decode_ms)
            if index >= 4
            and index != graph_build_step_index
            and index != profiled_decode_step
            and index != nsys_profiled_decode_step
        ]
        graph_setup_ms = [
            getattr(layer, "selector_graph_setup_ms", None) for layer in layers
        ]
        graph_builds = [
            getattr(layer, "selector_graph_builds", None) for layer in layers
        ]
        graph_setup_values = [v for v in graph_setup_ms if v is not None]
        graph_build_values = [v for v in graph_builds if v is not None]
        selector_abba_groups = {"eager": [], "graph": []}
        selector_abba_indices = {"eager": [], "graph": []}
        selector_abba_selector_groups = {"eager": [], "graph": []}
        if args.selector_abba:
            for index, elapsed in enumerate(decode_ms):
                if index < 4:
                    continue
                arm = "graph" if selector_abba_enabled_by_step[index] else "eager"
                selector_abba_groups[arm].append(elapsed)
                selector_abba_indices[arm].append(index)
                selector_abba_selector_groups[arm].append(
                    selector_abba_selector_ms_by_step[index]
                )

        def summarize_abba_arm(arm: str) -> dict:
            values = selector_abba_groups[arm]
            selector_values = selector_abba_selector_groups[arm]
            return {
                "steps": len(values),
                "step_indices": selector_abba_indices[arm],
                "decode_mean_ms": statistics.mean(values) if values else None,
                "decode_median_ms": statistics.median(values) if values else None,
                "selector_mean_ms_36_layers": (
                    statistics.mean(selector_values) if selector_values else None),
                "selector_median_ms_36_layers": (
                    statistics.median(selector_values) if selector_values else None),
            }

        def summarize_gather_arm(block: bool) -> dict:
            indices = [i for i, enabled in enumerate(gather_abba_enabled_by_step)
                       if i >= 4 and enabled == block]
            values = [decode_ms[i] for i in indices]
            gather_values = [gather_abba_ms_by_step[i] for i in indices]
            return {
                "steps": len(indices), "step_indices": indices,
                "decode_mean_ms": statistics.mean(values) if values else None,
                "decode_median_ms": statistics.median(values) if values else None,
                "gather_mean_ms_36_layers": statistics.mean(gather_values) if values else None,
                "gather_median_ms_36_layers": statistics.median(gather_values) if values else None,
            }

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
                "prompt_file": str(args.prompt_file) if args.prompt_file else None,
                "gen_tokens": args.gen_tokens,
                "index_select": bool(args.index_select), "chunk_size": 4096,
                "decode_kv_pipeline": args.kv_pipeline,
                "gather_abba": args.gather_abba,
                "selector_cuda_graph": args.selector_cuda_graph,
                "selector_static_mask": args.selector_static_mask,
                "fused_selector": args.fused_selector,
                "quant_history": args.quant_history,
                "gather_sort": args.gather_sort,
                "fused_dequant": args.fused_dequant,
                "selector_abba": args.selector_abba,
                "record_selected_ids": args.record_selected_ids,
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
                "steady_reporting_mean_ms": statistics.mean(steady_reporting)
                if steady_reporting else None,
                "steady_reporting_median_ms": statistics.median(steady_reporting)
                if steady_reporting else None,
                "steady_reporting_excluded_step_indices": (
                    [graph_build_step_index]
                    if graph_build_step_index is not None else []),
                "first_decode_step_ms": decode_ms[0] if decode_ms else None,
                "first_decode_step_includes_setup_costs": True,
            },
            "selector_graph": {
                "enabled": args.selector_cuda_graph,
                "setup_ms_by_layer": graph_setup_ms,
                "setup_ms_total": sum(graph_setup_values)
                if graph_setup_values else None,
                "graphs_built_by_layer": graph_builds,
                "graphs_built_total": sum(graph_build_values)
                if graph_build_values else None,
            },
            "selector_abba": {
                "enabled": args.selector_abba,
                "warmup_graph_steps": 4 if args.selector_abba else None,
                "steady_schedule": ["eager", "graph", "graph", "eager"]
                if args.selector_abba else None,
                "graph_enabled_by_decode_step": (
                    selector_abba_enabled_by_step if args.selector_abba else None),
                "selector_ms_by_decode_step_36_layers": (
                    selector_abba_selector_ms_by_step if args.selector_abba else None),
                "comparison_design": (
                    "same-process arms use different positions in one generated "
                    "trajectory; they are not same-query paired measurements"
                ) if args.selector_abba else None,
                "groups": {
                    "eager": summarize_abba_arm("eager"),
                    "graph": summarize_abba_arm("graph"),
                } if args.selector_abba else None,
            },
            "gather_abba": {
                "enabled": args.gather_abba,
                "warmup_pipeline_steps": 4 if args.gather_abba else None,
                "steady_schedule": ["serial", "pipeline", "pipeline", "serial"]
                if args.gather_abba else None,
                "pipeline_enabled_by_decode_step": gather_abba_enabled_by_step,
                "gather_ms_by_decode_step_36_layers": gather_abba_ms_by_step,
                "comparison_design": (
                    "same-process arms occupy different token positions; not same-query pairs"
                ) if args.gather_abba else None,
                "groups": {
                    "serial": summarize_gather_arm(False),
                    "pipeline": summarize_gather_arm(True),
                } if args.gather_abba else None,
            },
            "last_decode_stage_ms_36_layers": last_stage,
            "cuda_h2d_stage_event_spans_cpu_v_gather": args.kv_pipeline,
            "last_decode_cuda_event_ms_36_layers": cuda_stage,
            "profiling": {
                "trace_output": str(args.trace_output) if args.trace_output else None,
                "trace_step_index": profiled_decode_step,
                "nsys_trace_step_index": nsys_profiled_decode_step,
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
                "ids_history_by_layer": (
                    [layer.ids_history for layer in layers]
                    if args.record_selected_ids else None
                ),
            },
            "generated_token_ids": list(output[0][1]) if output else [],
            "runtime": {
                "layers": len(layers),
                "torch_cpu_threads": torch.get_num_threads(),
                "torch_interop_threads": torch.get_num_interop_threads(),
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
