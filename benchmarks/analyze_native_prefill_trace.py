#!/usr/bin/env python3
"""Attribute existing native prefill Chrome traces without rerunning the model.

This is a CPU-only parser for the 2026-10-07 closeout traces. CPU user-range
self time excludes the union of nested user_annotation ranges on the same
thread. CUDA kernels/copies/memsets are reported as interval unions and by
name; the asynchronous trace does not make range durations exclusive GPU time.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import traceback
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACE_ROOT = REPO_ROOT / "bench_logs/operator_bridge/closeout-profile-20261007"
DEFAULT_OUTPUT = DEFAULT_TRACE_ROOT / "native-prefill-trace-analysis.json"
WINDOWS = (
    ("flash", 4096, "flash/T4096-trace.json"),
    ("flash", 96, "flash/T96-trace.json"),
    ("flash_reuse", 4096, "flash_reuse/T4096-trace.json"),
    ("flash_reuse", 96, "flash_reuse/T96-trace.json"),
)
EXPECTED_ANNOTATIONS = (
    "m12.prefill_selector",
    "m12.selector_id_d2h",
    "m12.prefill_cpu_gather",
    "m12.prefill_h2d_pack",
    "m12.prefill_attention",
    "m12.prefill_store_kv",
)
GPU_ACTIVITY_CATEGORIES = {"kernel", "gpu_memcpy", "gpu_memset"}
HOST_LAUNCH_CATEGORIES = {"cuda_runtime", "cuda_driver"}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def event_interval(event: dict[str, Any]) -> tuple[float, float] | None:
    start = number(event.get("ts"))
    duration = number(event.get("dur"))
    if start is None or duration is None or duration < 0:
        return None
    return start, start + duration


def interval_union(intervals: Iterable[tuple[float, float]]) -> float:
    ordered = sorted((a, b) for a, b in intervals if b > a)
    if not ordered:
        return 0.0
    total = 0.0
    left, right = ordered[0]
    for start, end in ordered[1:]:
        if start > right:
            total += right - left
            left, right = start, end
        else:
            right = max(right, end)
    return total + right - left


def merge_intervals(intervals: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    ordered = sorted((a, b) for a, b in intervals if b > a)
    if not ordered:
        return []
    merged: list[tuple[float, float]] = []
    left, right = ordered[0]
    for start, end in ordered[1:]:
        if start > right:
            merged.append((left, right))
            left, right = start, end
        else:
            right = max(right, end)
    merged.append((left, right))
    return merged


def intersect_union(
    context: Iterable[tuple[float, float]],
    activity: Iterable[tuple[float, float]],
) -> float:
    contexts = merge_intervals(context)
    pieces: list[tuple[float, float]] = []
    for a, b in activity:
        for c, d in contexts:
            if d <= a:
                continue
            if c >= b:
                break
            left, right = max(a, c), min(b, d)
            if right > left:
                pieces.append((left, right))
    return interval_union(pieces)


def subtract_intervals(
    parent: tuple[float, float], children: Iterable[tuple[float, float]]
) -> list[tuple[float, float]]:
    start, end = parent
    result: list[tuple[float, float]] = []
    cursor = start
    for child_start, child_end in merge_intervals(children):
        left, right = max(start, child_start), min(end, child_end)
        if right <= left:
            continue
        if left > cursor:
            result.append((cursor, left))
        cursor = max(cursor, right)
    if cursor < end:
        result.append((cursor, end))
    return result


def numeric_summary(values: list[float]) -> dict[str, float | None]:
    return {
        "sum_us": sum(values),
        "mean_us": statistics.fmean(values) if values else None,
        "median_us": statistics.median(values) if values else None,
        "min_us": min(values) if values else None,
        "max_us": max(values) if values else None,
    }


def valid_complete_events(
    events: list[dict[str, Any]], predicate: Any
) -> list[tuple[int, dict[str, Any], tuple[float, float]]]:
    rows = []
    for index, event in enumerate(events):
        interval = event_interval(event)
        if interval is not None and event.get("ph") == "X" and predicate(event):
            rows.append((index, event, interval))
    return rows


def annotation_name(name: Any) -> bool:
    return isinstance(name, str) and (
        name.startswith("m12.") or name.startswith("nanokv.profile_step.")
    )


def event_key(event: dict[str, Any]) -> tuple[Any, Any]:
    return event.get("pid"), event.get("tid")


def arg_value(event: dict[str, Any], key: str) -> Any:
    args = event.get("args")
    return args.get(key) if isinstance(args, dict) else None


def gpu_activity_groups(
    rows: list[tuple[int, dict[str, Any], tuple[float, float]]]
) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[tuple[int, dict[str, Any], tuple[float, float]]]] = defaultdict(list)
    by_category: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        _, event, interval = row
        category = str(event.get("cat"))
        name = str(event.get("name", "<unnamed>"))
        groups[(category, name)].append(row)
        by_category[category].append(interval)

    names = []
    for (category, name), group in groups.items():
        intervals = [row[2] for row in group]
        bytes_values = [number(arg_value(row[1], "bytes")) for row in group]
        byte_values = [value for value in bytes_values if value is not None]
        durations = [b - a for a, b in intervals]
        names.append(
            {
                "category": category,
                "name": name,
                "calls": len(group),
                "duration_sum_us": sum(durations),
                "duration_stats": numeric_summary(durations),
                "interval_union_us": interval_union(intervals),
                "bytes_sum": sum(byte_values) if byte_values else None,
            }
        )
    names.sort(key=lambda row: (-row["interval_union_us"], -row["duration_sum_us"], row["name"]))
    return {
        "activity_count": len(rows),
        "interval_union_us": interval_union(row[2] for row in rows),
        "activity_span_us": (
            max(row[2][1] for row in rows) - min(row[2][0] for row in rows) if rows else 0.0
        ),
        "by_category": {
            category: {
                "calls": sum(1 for row in rows if row[1].get("cat") == category),
                "interval_union_us": interval_union(intervals),
            }
            for category, intervals in sorted(by_category.items())
        },
        "by_name": names,
    }


def annotation_context_groups(
    rows: list[tuple[int, dict[str, Any], tuple[float, float]]],
    activity_rows: list[tuple[int, dict[str, Any], tuple[float, float]]],
) -> dict[str, Any]:
    groups: dict[str, list[tuple[int, dict[str, Any], tuple[float, float]]]] = defaultdict(list)
    for row in rows:
        groups[str(row[1].get("name", "<unnamed>"))].append(row)
    activities = [row[2] for row in activity_rows]
    result = {}
    for name, group in sorted(groups.items()):
        contexts = [row[2] for row in group]
        result[name] = {
            "calls": len(group),
            "annotation_duration_sum_us": sum(b - a for a, b in contexts),
            "annotation_interval_union_us": interval_union(contexts),
            "overlapping_gpu_activity_count": sum(
                any(a < d and c < b for c, d in contexts) for a, b in activities
            ),
            "gpu_activity_intersection_union_us": intersect_union(contexts, activities),
            "interpretation": "GPU-side labeled range context; duration is not exclusive kernel time.",
        }
    return result


def trace_flow_ids(events: list[dict[str, Any]]) -> dict[str, set[Any]]:
    starts: set[Any] = set()
    finishes: set[Any] = set()
    for event in events:
        if event.get("cat") != "ac2g":
            continue
        if event.get("ph") == "s":
            starts.add(event.get("id"))
        elif event.get("ph") == "f":
            finishes.add(event.get("id"))
    return {"starts": starts, "finishes": finishes, "pairs": starts & finishes}


def correlations_for_annotation(
    annotation: dict[str, Any],
    interval: tuple[float, float],
    api_rows: list[tuple[int, dict[str, Any], tuple[float, float]]],
    gpu_by_correlation: dict[Any, list[tuple[int, dict[str, Any], tuple[float, float]]]],
    cpu_op_rows: list[tuple[int, dict[str, Any], tuple[float, float]]],
    gpu_by_external_id: dict[Any, list[tuple[int, dict[str, Any], tuple[float, float]]]],
) -> dict[str, Any]:
    annotation_pid_tid = event_key(annotation)
    start, end = interval
    runtime_correlated: dict[int, tuple[int, dict[str, Any], tuple[float, float]]] = {}
    cpu_op_correlated: dict[int, tuple[int, dict[str, Any], tuple[float, float]]] = {}
    api_calls = 0
    cpu_ops = 0

    for _, host_event, host_interval in api_rows:
        if event_key(host_event) != annotation_pid_tid:
            continue
        if not (start <= host_interval[0] < end):
            continue
        api_calls += 1
        correlation = arg_value(host_event, "correlation")
        for gpu_row in gpu_by_correlation.get(correlation, ()):
            runtime_correlated[gpu_row[0]] = gpu_row

    for _, cpu_op, cpu_op_interval in cpu_op_rows:
        if event_key(cpu_op) != annotation_pid_tid:
            continue
        if not (start <= cpu_op_interval[0] < end):
            continue
        cpu_ops += 1
        external_id = arg_value(cpu_op, "External id")
        for gpu_row in gpu_by_external_id.get(external_id, ()):
            cpu_op_correlated[gpu_row[0]] = gpu_row

    union_rows = {**runtime_correlated, **cpu_op_correlated}
    activities = list(union_rows.values())
    return {
        "host_runtime_or_driver_api_calls_started_inside": api_calls,
        "cpu_op_calls_started_inside": cpu_ops,
        "runtime_correlation_linked_gpu_activity_count": len(runtime_correlated),
        "cpu_op_external_id_linked_gpu_activity_count": len(cpu_op_correlated),
        "unique_correlated_gpu_activity_count": len(union_rows),
        "correlated_gpu_activity_interval_union_us": interval_union(row[2] for row in activities),
        "correlated_gpu_activity_by_name": grouped_activity_rows(activities),
    }


def grouped_activity_rows(
    rows: list[tuple[int, dict[str, Any], tuple[float, float]]]
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[tuple[int, dict[str, Any], tuple[float, float]]]] = defaultdict(list)
    for row in rows:
        groups[(str(row[1].get("cat")), str(row[1].get("name", "<unnamed>")))].append(row)
    result = []
    for (category, name), group in groups.items():
        durations = [row[2][1] - row[2][0] for row in group]
        result.append(
            {
                "category": category,
                "name": name,
                "calls": len(group),
                "duration_sum_us": sum(durations),
                "interval_union_us": interval_union(row[2] for row in group),
            }
        )
    return sorted(result, key=lambda row: (-row["interval_union_us"], row["name"]))


def host_api_summary(rows: list[tuple[int, dict[str, Any], tuple[float, float]]]) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[tuple[int, dict[str, Any], tuple[float, float]]]] = defaultdict(list)
    for row in rows:
        groups[(str(row[1].get("cat")), str(row[1].get("name", "<unnamed>")))].append(row)
    by_name = []
    for (category, name), group in groups.items():
        intervals = [row[2] for row in group]
        by_name.append(
            {
                "category": category,
                "name": name,
                "calls": len(group),
                "duration_sum_us": sum(b - a for a, b in intervals),
                "interval_union_us": interval_union(intervals),
            }
        )
    by_name.sort(key=lambda row: (-row["interval_union_us"], row["name"]))
    return {
        "calls": len(rows),
        "interval_union_us": interval_union(row[2] for row in rows),
        "by_name": by_name,
    }


def analyze_trace(path: Path, backend: str, tokens: int) -> dict[str, Any]:
    raw = path.read_bytes()
    document = json.loads(raw)
    if not isinstance(document, dict) or not isinstance(document.get("traceEvents"), list):
        raise ValueError("expected a Chrome trace object with a traceEvents list")
    events = document["traceEvents"]
    if not all(isinstance(event, dict) for event in events):
        raise ValueError("traceEvents contains a non-object record")

    user_rows = valid_complete_events(
        events,
        lambda event: event.get("cat") == "user_annotation" and annotation_name(event.get("name")),
    )
    gpu_rows = valid_complete_events(events, lambda event: event.get("cat") in GPU_ACTIVITY_CATEGORIES)
    gpu_annotation_rows = valid_complete_events(
        events,
        lambda event: event.get("cat") == "gpu_user_annotation" and annotation_name(event.get("name")),
    )
    api_rows = valid_complete_events(events, lambda event: event.get("cat") in HOST_LAUNCH_CATEGORIES)
    cpu_op_rows = valid_complete_events(events, lambda event: event.get("cat") == "cpu_op")

    if not user_rows:
        raise ValueError("no complete CPU user_annotation X ranges were found")
    if not gpu_rows:
        raise ValueError("no complete CUDA kernel/copy/memset X activities were found")

    expected_step_name = f"nanokv.profile_step.T{tokens}"
    counts = Counter(row[1].get("name") for row in user_rows)
    missing = [name for name in EXPECTED_ANNOTATIONS if counts.get(name, 0) != 36]
    step_count = counts.get(expected_step_name, 0)
    if missing or step_count != 1:
        raise ValueError(
            f"unexpected annotation coverage: expected each M12 stage 36 times and {expected_step_name} once; "
            f"bad_stages={missing}, step_calls={step_count}"
        )

    gpu_by_correlation: dict[Any, list[tuple[int, dict[str, Any], tuple[float, float]]]] = defaultdict(list)
    gpu_by_external_id: dict[Any, list[tuple[int, dict[str, Any], tuple[float, float]]]] = defaultdict(list)
    for row in gpu_rows:
        correlation = arg_value(row[1], "correlation")
        external_id = arg_value(row[1], "External id")
        if correlation is not None:
            gpu_by_correlation[correlation].append(row)
        if external_id is not None:
            gpu_by_external_id[external_id].append(row)

    cpu_op_external_ids = {arg_value(row[1], "External id") for row in cpu_op_rows}
    host_correlations = {
        arg_value(row[1], "correlation") for row in api_rows if arg_value(row[1], "correlation") is not None
    }
    ext_matches = sum(arg_value(row[1], "External id") in cpu_op_external_ids for row in gpu_rows)
    corr_matches = sum(arg_value(row[1], "correlation") in host_correlations for row in gpu_rows)
    flows = trace_flow_ids(events)

    rows_by_thread: dict[tuple[Any, Any], list[tuple[int, dict[str, Any], tuple[float, float]]]] = defaultdict(list)
    for row in user_rows:
        rows_by_thread[event_key(row[1])].append(row)

    annotations: dict[str, list[dict[str, Any]]] = defaultdict(list)
    annotation_instances: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row_index, event, parent_interval in user_rows:
        children = []
        for child_index, child_event, child_interval in rows_by_thread[event_key(event)]:
            if child_index == row_index:
                continue
            if parent_interval[0] <= child_interval[0] and child_interval[1] <= parent_interval[1]:
                children.append(child_interval)
        self_intervals = subtract_intervals(parent_interval, children)
        direct_gpu = [
            gpu_row for gpu_row in gpu_rows
            if parent_interval[0] < gpu_row[2][1] and gpu_row[2][0] < parent_interval[1]
        ]
        context = correlations_for_annotation(
            event,
            parent_interval,
            api_rows,
            gpu_by_correlation,
            cpu_op_rows,
            gpu_by_external_id,
        )
        name = str(event.get("name"))
        duration = parent_interval[1] - parent_interval[0]
        item = {
            "ts_us": parent_interval[0],
            "cpu_inclusive_us": duration,
            "nested_user_annotation_union_us": interval_union(children),
            "cpu_self_excluding_nested_annotations_us": interval_union(self_intervals),
            "gpu_activity_timeline_overlap_count": len({gpu_row[0] for gpu_row in direct_gpu}),
            "gpu_activity_timeline_overlap_union_us": intersect_union(
                [parent_interval], (gpu_row[2] for gpu_row in direct_gpu)
            ),
            "correlation_context": context,
        }
        annotations[name].append(
            {
                "event": event,
                "interval": parent_interval,
                "children": children,
                "self_intervals": self_intervals,
                "direct_gpu": direct_gpu,
                "correlation_context": context,
            }
        )
        annotation_instances[name].append(item)

    annotation_summary = {}
    for name, instances in sorted(annotations.items()):
        intervals = [instance["interval"] for instance in instances]
        self_intervals = [part for instance in instances for part in instance["self_intervals"]]
        inclusive_values = [b - a for a, b in intervals]
        direct_gpu_indexes = {
            row[0] for instance in instances for row in instance["direct_gpu"]
        }
        correlated_gpu_indexes = set()
        by_correlated_name: dict[tuple[str, str], list[tuple[int, dict[str, Any], tuple[float, float]]]] = defaultdict(list)
        for instance in instances:
            event = instance["event"]
            start, end = instance["interval"]
            thread = event_key(event)
            runtime_ids = set()
            cpu_ids = set()
            for _, api_event, api_interval in api_rows:
                if event_key(api_event) == thread and start <= api_interval[0] < end:
                    runtime_ids.update(
                        row[0] for row in gpu_by_correlation.get(arg_value(api_event, "correlation"), ())
                    )
            for _, cpu_event, cpu_interval in cpu_op_rows:
                if event_key(cpu_event) == thread and start <= cpu_interval[0] < end:
                    cpu_ids.update(
                        row[0] for row in gpu_by_external_id.get(arg_value(cpu_event, "External id"), ())
                    )
            correlated_gpu_indexes.update(runtime_ids | cpu_ids)
        for index, activity, interval in gpu_rows:
            if index in correlated_gpu_indexes:
                by_correlated_name[(str(activity.get("cat")), str(activity.get("name", "<unnamed>")))].append(
                    (index, activity, interval)
                )
        annotation_summary[name] = {
            "calls": len(instances),
            "cpu_inclusive": numeric_summary(inclusive_values),
            "cpu_inclusive_interval_union_us": interval_union(intervals),
            "cpu_self_excluding_nested_annotations": numeric_summary(
                [instance["cpu_self_excluding_nested_annotations_us"] for instance in annotation_instances[name]]
            ),
            "cpu_self_interval_union_us": interval_union(self_intervals),
            "gpu_timeline_overlap_activity_count": len(direct_gpu_indexes),
            "gpu_timeline_overlap_union_us": intersect_union(
                intervals, (row[2] for row in gpu_rows if row[0] in direct_gpu_indexes)
            ),
            "correlated_gpu_activity_count": len(correlated_gpu_indexes),
            "correlated_gpu_activity_by_name": grouped_activity_rows(
                [row for group in by_correlated_name.values() for row in group]
            ),
            "instances": annotation_instances[name],
        }

    step_instance = next(
        instance for instance in annotations[expected_step_name]
    )
    stage_signals = [
        {
            "name": name,
            "cpu_self_interval_union_us": annotation_summary.get(name, {}).get("cpu_self_interval_union_us", 0.0),
            "cpu_inclusive_interval_union_us": annotation_summary.get(name, {}).get("cpu_inclusive_interval_union_us", 0.0),
            "correlated_gpu_activity_count": annotation_summary.get(name, {}).get("correlated_gpu_activity_count", 0),
        }
        for name in EXPECTED_ANNOTATIONS
    ]
    stage_signals.sort(key=lambda row: (-row["cpu_self_interval_union_us"], row["name"]))

    gpu_annotation_summary = annotation_context_groups(gpu_annotation_rows, gpu_rows)
    host_summary = host_api_summary(api_rows)
    profile_step_duration = step_instance["interval"][1] - step_instance["interval"][0]
    flow_linked_activities = sum(
        arg_value(row[1], "correlation") in flows["pairs"] for row in gpu_rows
    )

    return {
        "backend": backend,
        "tokens": tokens,
        "trace_path": str(path.resolve()),
        "trace_sha256": sha256_bytes(raw),
        "trace_bytes": len(raw),
        "trace_id": document.get("trace_id"),
        "trace_metadata": {
            key: document.get(key)
            for key in ("schemaVersion", "displayTimeUnit", "baseTimeNanoseconds", "cuda_runtime_version", "cuda_driver_version")
        },
        "trace_timing_unit": "microseconds for Chrome trace ts/dur values; displayTimeUnit is retained as source metadata",
        "event_count": len(events),
        "event_counts_by_category": dict(sorted(Counter(str(event.get("cat", "<none>")) for event in events).items())),
        "step_range": {
            "name": expected_step_name,
            "calls": 1,
            "cpu_inclusive_us": profile_step_duration,
            "cpu_self_excluding_nested_annotations_us": interval_union(step_instance["self_intervals"]),
            "interpretation": "Instrumented llm.step user range; includes CPU-side waits and profiler overhead, not an uninstrumented TTFT result.",
        },
        "cpu_user_annotations": annotation_summary,
        "gpu_activities": gpu_activity_groups(gpu_rows),
        "gpu_user_annotation_contexts": gpu_annotation_summary,
        "cuda_host_api": host_summary,
        "correlation_coverage": {
            "gpu_activity_count": len(gpu_rows),
            "activities_matching_cpu_op_external_id": ext_matches,
            "activities_matching_host_api_correlation": corr_matches,
            "activities_with_ac2g_start_and_finish_for_correlation": flow_linked_activities,
            "ac2g_flow_ids": {
                "start_count": len(flows["starts"]),
                "finish_count": len(flows["finishes"]),
                "paired_id_count": len(flows["pairs"]),
            },
            "method": "GPU args.External id is matched to cpu_op args.External id; GPU args.correlation is matched to cuda_runtime/cuda_driver args.correlation. ac2g flow start/finish IDs are recorded as corroborating trace context.",
            "limitation": "Correlation links captured activity to a CPU op or launch API; it does not prove exclusive stage cost, critical-path impact, or that overlapping GPU work was caused by only one enclosing annotation.",
        },
        "cpu_annotation_self_time_definition": "For each user_annotation X range, subtract the interval union of all fully contained user_annotation X ranges on the same pid/tid. This is structural annotation self time, not sampled CPU utilization or proof that the thread was not waiting.",
        "diagnostic_signals": {
            "user_objective": "Improve first-token latency while keeping sparse settings fixed and generated output consistent.",
            "cpu_annotation_self_time_order": stage_signals,
            "interpretation": "Candidate investigation order only. Annotation self time, inclusive range time, GPU interval union, and correlation-linked activity are separate views and must not be summed across stages as exclusive TTFT shares.",
            "recommendations": [
                "For the fixed sparse settings and output-consistency requirement, inspect per-call selector, CPU-gather, H2D-pack, attention, and KV-store ranges together with their correlated activities; use CPU self time to distinguish nested annotation overhead, and keep CPU inclusive time as a wait-inclusive bound.",
                "Treat GPU-side m12 ranges as labels over device timeline context. Compare their overlap with actual kernel/copy intervals; never report their NVTX duration as exclusive GPU execution time.",
                "These are instrumented T4096/T96 windows for flash and flash_reuse only. Choose a candidate from the trace, then verify first-token latency in matched fresh-process uninstrumented runs with identical inputs/settings and exact-output checks.",
                "The operator backend has no closeout trace, so this artifact cannot compare its stage attribution. The existing adapter report also records selector-set differences after the 96-token tail; preserve that quality finding when evaluating output consistency.",
            ],
            "materials": [
                "bench_logs/operator_bridge/closeout-profile-20261007/flash/T4096-trace.json",
                "bench_logs/operator_bridge/closeout-profile-20261007/flash/T96-trace.json",
                "bench_logs/operator_bridge/closeout-profile-20261007/flash_reuse/T4096-trace.json",
                "bench_logs/operator_bridge/closeout-profile-20261007/flash_reuse/T96-trace.json",
                "docs/NATIVE_ENGINE_CLOSEOUT_PROFILE_PLAN.md",
                "docs/NATIVE_ENGINE_OPTIMIZATION_NEXT_SCOPE.md",
                "docs/NATIVE_SEGMENTED_ADAPTER_REPORT.md",
                "benchmarks/profile_segmented_adapter.py",
            ],
        },
    }


def source_hashes(trace_paths: list[Path]) -> dict[str, Any]:
    named = {
        "analyzer_script": Path(__file__).resolve(),
        "reference_wrapper": REPO_ROOT / "benchmarks/profile_segmented_adapter.py",
        "closeout_profile_plan": REPO_ROOT / "docs/NATIVE_ENGINE_CLOSEOUT_PROFILE_PLAN.md",
        "optimization_next_scope": REPO_ROOT / "docs/NATIVE_ENGINE_OPTIMIZATION_NEXT_SCOPE.md",
    }
    hashes: dict[str, Any] = {}
    for label, path in named.items():
        hashes[label] = {
            "path": str(path),
            "sha256": sha256_file(path) if path.is_file() else None,
            "exists": path.is_file(),
        }
    hashes["traces"] = {
        str(path.resolve()): {
            "sha256": sha256_file(path) if path.is_file() else None,
            "bytes": path.stat().st_size if path.is_file() else None,
            "exists": path.is_file(),
        }
        for path in trace_paths
    }
    return hashes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace-root",
        type=Path,
        default=DEFAULT_TRACE_ROOT,
        help="directory containing flash/ and flash_reuse/ trace folders",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="new JSON report path; existing files are never overwritten",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    trace_root = args.trace_root.resolve()
    output = args.output.resolve()
    trace_paths = [trace_root / relative for _, _, relative in WINDOWS]
    record: dict[str, Any] = {
        "schema": "nanokv.native_prefill_trace_analysis.v1",
        "status": "running",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_mode": "offline CPU-only Chrome trace parsing; no model, CUDA, benchmark, or runtime was executed",
        "trace_root": str(trace_root),
        "output_path": str(output),
        "source_hashes": {},
        "windows": [],
        "failures": [],
        "measurement_scope": "Existing instrumented flash and flash_reuse archive prompt windows T4096 and T96; operator profile absent. Not an uninstrumented TTFT comparison or quality result.",
    }

    try:
        if output.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {output}")
        record["source_hashes"] = source_hashes(trace_paths)
        for backend, tokens, relative in WINDOWS:
            path = trace_root / relative
            try:
                if not path.is_file():
                    raise FileNotFoundError(f"trace file does not exist: {path}")
                record["windows"].append(analyze_trace(path, backend, tokens))
            except Exception as exc:
                record["failures"].append(
                    {
                        "backend": backend,
                        "tokens": tokens,
                        "trace_path": str(path.resolve()),
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    }
                )
        record["status"] = "passed" if not record["failures"] and len(record["windows"]) == len(WINDOWS) else "failed"
    except Exception as exc:
        record["status"] = "failed"
        record["failures"].append(
            {"scope": "setup", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
        )
        # An existing report must remain untouched. In that case, print the
        # refusal and return without creating a second record elsewhere.
        if isinstance(exc, FileExistsError):
            print(f"analysis failed; {exc}", file=sys.stderr)
            return 2

    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
    except FileExistsError:
        print(f"analysis failed; refusing to overwrite existing output: {output}", file=sys.stderr)
        return 2

    print(f"analysis {record['status']}: {output}")
    if record["failures"]:
        print(f"failures recorded: {len(record['failures'])}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
