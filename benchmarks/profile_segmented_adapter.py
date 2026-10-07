"""Diagnostic wrapper; does not modify the frozen runtime or performance driver."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import traceback

import torch
import benchmark_segmented_adapter as driver


def trace_summary(path):
    events = json.loads(path.read_text())["traceEvents"]
    gpu = [e for e in events if e.get("ph") == "X" and
           e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    if not gpu:
        raise RuntimeError("no CUDA activities captured")
    groups = defaultdict(lambda: dict(calls=0, activity_total_us=0.))
    intervals = []
    for event in gpu:
        key = event["cat"] + ": " + event["name"]
        groups[key]["calls"] += 1
        groups[key]["activity_total_us"] += event["dur"]
        intervals.append((event["ts"], event["ts"] + event["dur"]))
    intervals.sort()
    start, end = intervals[0]
    union = 0.
    for a, b in intervals[1:]:
        if a > end:
            union += end-start
            start, end = a, b
        else:
            end = max(end, b)
    union += end-start
    return dict(gpu_activities=len(gpu), gpu_activity_interval_union_us=union,
                gpu_activity_span_us=max(b for _, b in intervals)-intervals[0][0],
                activities={k: v for k, v in sorted(groups.items(), key=lambda kv: -kv[1]["activity_total_us"])})


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--operator-root", type=Path, required=True)
    parser.add_argument("--backend", choices=("flash", "flash_reuse", "operator"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    names = ("benchmarks/profile_segmented_adapter.py", "docs/NATIVE_ENGINE_CLOSEOUT_PROFILE_PLAN.md",
             "benchmarks/benchmark_segmented_adapter.py")
    hashes = {n: driver.sha(driver.ROOT/n) for n in names}
    record = dict(status="running", backend=args.backend, source_hashes=hashes, windows=[],
                  measurement_usable_as_clean_baseline=False,
                  scope="single instrumented archive main/tail diagnostic, not performance A/B or hardware counters")
    output = args.output_dir/"profile-summary.json"
    raw = args.output_dir/"instrumented-driver.json"
    original_step = driver.LLM.step
    calls = 0

    def step(llm):
        nonlocal calls
        call = calls
        calls += 1
        # Five warm steps precede five archive steps; code is not profiled.
        if call not in (8, 9):
            return original_step(llm)
        tokens = 4096 if call == 8 else 96
        layers = [m.sparse_rt for m in llm.model_runner.model.modules() if getattr(m, "sparse_rt", None)]
        if len(layers) != 36:
            raise RuntimeError("expected 36 layers")
        expected_prior = 12288 if call == 8 else 16384
        if any(rt.valid_len != expected_prior for rt in layers):
            raise RuntimeError("profile window shifted")
        for rt in layers:
            rt.profile_torch_stages = True
        try:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                   torch.profiler.ProfilerActivity.CUDA]) as prof:
                with torch.profiler.record_function(f"nanokv.profile_step.T{tokens}"):
                    result = original_step(llm)
                    torch.cuda.synchronize()
        finally:
            for rt in layers:
                rt.profile_torch_stages = False
        trace = args.output_dir/f"T{tokens}-trace.json"
        prof.export_chrome_trace(str(trace))
        ranges = {}
        for event in prof.key_averages():
            if event.key.startswith(("m12.", "nanokv.profile_step.")):
                ranges[event.key] = dict(calls=event.count, cpu_inclusive_us=event.cpu_time_total,
                                         cpu_self_us=event.self_cpu_time_total,
                                         device_inclusive_us=event.device_time_total)
        for name in ("prefill_selector", "prefill_cpu_gather", "prefill_h2d_pack", "prefill_attention", "prefill_store_kv"):
            if ranges["m12."+name]["calls"] != 36:
                raise RuntimeError("incomplete stage ranges: "+name)
        record["windows"].append(dict(tokens=tokens, prior_tokens=expected_prior,
                                       trace_name=trace.name, trace_sha256=driver.sha(trace),
                                       trace_bytes=trace.stat().st_size, ranges=ranges, **trace_summary(trace)))
        output.write_text(json.dumps(record, indent=2)+"\n")
        return result

    driver.LLM.step = step
    old_argv = sys.argv
    try:
        sys.argv = [str(driver.ROOT/"benchmarks/benchmark_segmented_adapter.py"),
                    "--model", str(args.model), "--manifest", str(args.manifest),
                    "--operator-root", str(args.operator_root), "--backend", args.backend,
                    "--phase", "perf", "--output", str(raw)]
        driver.run()
        child = json.loads(raw.read_text())
        if child["status"] != "passed" or len(record["windows"]) != 2 or calls != 15:
            raise RuntimeError("incomplete diagnostic request/window sequence")
        if [r["generated_ids"] for r in child["requests"]] != [[2797], [57912]]:
            raise RuntimeError("first generated IDs differ from original captures")
        after = {n: driver.sha(driver.ROOT/n) for n in names}
        if after != hashes:
            raise RuntimeError("profile wrapper source changed during run")
        record.update(status="passed", source_hashes_after=after,
                      driver_record_sha256=driver.sha(raw), driver_record_name=raw.name,
                      runtime=child["runtime"], manifest_sha256=child["manifest_sha256"],
                      generated_ids=[r["generated_ids"] for r in child["requests"]],
                      native_sources_verified=len(child["source_hashes"]))
    except Exception as exc:
        record.update(status="failed", failure=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        driver.LLM.step = original_step
        sys.argv = old_argv
        output.write_text(json.dumps(record, indent=2)+"\n")
        print("diagnostic profile", output, flush=True)


if __name__ == "__main__":
    run()
