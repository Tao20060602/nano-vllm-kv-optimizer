"""Compare single decode Nsight traces without claiming GPU utilization."""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path


NS_PER_MS = 1_000_000
STEP_LABEL = "nanokv.decode.step"
DEVICE_TABLES = (
    ("CUPTI_ACTIVITY_KIND_KERNEL", "kernel"),
    ("CUPTI_ACTIVITY_KIND_MEMCPY", "memcpy"),
    ("CUPTI_ACTIVITY_KIND_MEMSET", "memset"),
)
STAGE_NAMES = (
    "cpu_gather", "selector", "selector_id_d2h", "h2d_pack",
    "packed_attention", "recent_update",
)
COPY_KIND_FALLBACK = {
    0: "UNKNOWN", 1: "HOST_TO_DEVICE", 2: "DEVICE_TO_HOST",
    3: "DEVICE_TO_DEVICE", 4: "DEVICE_TO_HOST_STAGED",
    5: "HOST_TO_DEVICE_STAGED", 6: "DEFAULT", 7: "PEER_TO_PEER",
}


def merge_intervals(intervals, clip_start=None, clip_end=None):
    """Return the union as sorted half-open intervals, optionally clipped."""
    prepared = []
    for start, end in intervals:
        start, end = int(start), int(end)
        if clip_start is not None:
            start = max(start, int(clip_start))
        if clip_end is not None:
            end = min(end, int(clip_end))
        if end > start:
            prepared.append((start, end))
    prepared.sort()
    merged = []
    for start, end in prepared:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def union_duration(intervals, clip_start=None, clip_end=None):
    return sum(end - start for start, end in
               merge_intervals(intervals, clip_start, clip_end))


def _quote(identifier):
    return '"' + identifier.replace('"', '""') + '"'


def _tables(connection):
    return {row[0].lower(): row[0] for row in connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}


def _columns(connection, table):
    return {row[1].lower(): row[1] for row in connection.execute(
        f"PRAGMA table_info({_quote(table)})")}


def _column(columns, *choices):
    for name in choices:
        if name.lower() in columns:
            return columns[name.lower()]
    return None


def _string_ids(connection, tables):
    table = tables.get("stringids")
    if table is None:
        return {}
    columns = _columns(connection, table)
    key_col = _column(columns, "id")
    value_col = _column(columns, "value", "name", "text")
    if key_col is None or value_col is None:
        return {}
    return dict(connection.execute(
        f"SELECT {_quote(key_col)}, {_quote(value_col)} FROM {_quote(table)}"))


def _enum_values(connection, tables, table_name):
    table = tables.get(table_name.lower())
    if table is None:
        return {}
    columns = _columns(connection, table)
    key_col = _column(columns, "id", "value", "enumValue")
    value_col = _column(columns, "name", "value", "label", "description")
    if key_col is None or value_col is None or key_col == value_col:
        return {}
    return dict(connection.execute(
        f"SELECT {_quote(key_col)}, {_quote(value_col)} FROM {_quote(table)}"))


def _resolve_text(value, strings):
    if value is None:
        return ""
    return str(strings.get(value, value))


def _resolve_copy_kind(value, enum_values):
    if value is None:
        return "UNKNOWN"
    resolved = enum_values.get(value)
    if resolved is not None:
        return str(resolved)
    try:
        return COPY_KIND_FALLBACK.get(int(value), str(value))
    except (TypeError, ValueError):
        return str(value)


def _is_h2d(raw_kind, label):
    text = str(label).upper().replace("-", "_").replace(" ", "_")
    if "HOST_TO_DEVICE" in text or "HTOD" in text or "H2D" in text:
        return True
    try:
        return int(raw_kind) in (1, 5)
    except (TypeError, ValueError):
        return False


def _connect_readonly(path):
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    connection = sqlite3.connect(resolved.as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection, resolved


def _load_nvtx(connection, tables, strings):
    table = tables.get("nvtx_events")
    if table is None:
        raise RuntimeError("trace has no NVTX_EVENTS table")
    columns = _columns(connection, table)
    start_col = _column(columns, "start")
    end_col = _column(columns, "end")
    tid_col = _column(columns, "globalTid")
    text_col = _column(columns, "text")
    text_id_col = _column(columns, "textId")
    if start_col is None or end_col is None or (text_col is None and text_id_col is None):
        raise RuntimeError("NVTX_EVENTS lacks start/end/text or textId columns")
    selects = [f"{_quote(start_col)} AS start", f"{_quote(end_col)} AS end"]
    selects.append(f"{_quote(tid_col)} AS globalTid" if tid_col else "NULL AS globalTid")
    selects.append(f"{_quote(text_col)} AS text" if text_col else "NULL AS text")
    selects.append(f"{_quote(text_id_col)} AS textId" if text_id_col else "NULL AS textId")
    rows = connection.execute(
        f"SELECT {', '.join(selects)} FROM {_quote(table)}").fetchall()
    events = []
    for row in rows:
        if row["start"] is None or row["end"] is None:
            continue
        text = row["text"]
        if text is None or text == "":
            text = _resolve_text(row["textId"], strings)
        events.append({
            "start": int(row["start"]), "end": int(row["end"]),
            "tid": row["globalTid"], "text": str(text or ""),
        })
    return events


def _stage_name(text):
    lowered = text.lower()
    if "selector_id_d2h" in lowered:
        return "selector_id_d2h"
    if re.search(r"(?:^|\.)selector(?:\.|$)", lowered):
        return "selector"
    for name in ("cpu_gather", "h2d_pack", "packed_attention", "recent_update"):
        if name in lowered:
            return name
    return None


def _device_events(connection, tables, window_start, window_end):
    events, missing_tables = [], []
    for table_name, kind in DEVICE_TABLES:
        table = tables.get(table_name.lower())
        if table is None:
            missing_tables.append(table_name)
            continue
        columns = _columns(connection, table)
        start_col = _column(columns, "start")
        end_col = _column(columns, "end")
        if start_col is None or end_col is None:
            missing_tables.append(table_name + "(missing start/end)")
            continue
        fields = {
            "bytes": _column(columns, "bytes", "size"),
            "copy_kind": _column(columns, "copyKind", "kind"),
            "correlation_id": _column(columns, "correlationId"),
            "global_pid": _column(columns, "globalPid"),
        }
        select_fields = [f"{_quote(start_col)} AS start", f"{_quote(end_col)} AS end"]
        for alias, column in fields.items():
            select_fields.append(f"{_quote(column)} AS {_quote(alias)}" if column
                                 else f"NULL AS {_quote(alias)}")
        rows = connection.execute(
            f"SELECT {', '.join(select_fields)} FROM {_quote(table)} "
            f"WHERE {_quote(end_col)} > ? AND {_quote(start_col)} < ?",
            (window_start, window_end)).fetchall()
        for row in rows:
            start, end = int(row["start"]), int(row["end"])
            clipped_start, clipped_end = max(start, window_start), min(end, window_end)
            if clipped_end <= clipped_start:
                continue
            events.append({
                "kind": kind, "table": table, "start": start, "end": end,
                "clip_start": clipped_start, "clip_end": clipped_end,
                "bytes": row["bytes"], "copy_kind": row["copy_kind"],
                "correlation_id": row["correlation_id"],
                "global_pid": row["global_pid"],
            })
    return events, missing_tables


def _load_cuda_apis(connection, tables, strings, window, device_events):
    table = tables.get("cupti_activity_kind_runtime")
    if table is None:
        return [], "CUPTI_ACTIVITY_KIND_RUNTIME missing"
    columns = _columns(connection, table)
    start_col = _column(columns, "start")
    end_col = _column(columns, "end")
    tid_col = _column(columns, "globalTid")
    name_col = _column(columns, "nameId", "textId", "name")
    corr_col = _column(columns, "correlationId")
    class_col = _column(columns, "eventClass")
    if start_col is None or end_col is None or name_col is None:
        return [], "CUPTI_ACTIVITY_KIND_RUNTIME lacks time/name columns"
    select_fields = [f"{_quote(start_col)} AS start", f"{_quote(end_col)} AS end",
                     f"{_quote(name_col)} AS name_id"]
    for alias, column in (("tid", tid_col), ("correlation_id", corr_col),
                          ("event_class", class_col)):
        select_fields.append(f"{_quote(column)} AS {_quote(alias)}" if column
                             else f"NULL AS {_quote(alias)}")
    where = [f"{_quote(start_col)} >= ?", f"{_quote(end_col)} <= ?"]
    params = [window["start"], window["end"]]
    if tid_col and window["tid"] is not None:
        where.append(f"{_quote(tid_col)} = ?")
        params.append(window["tid"])
    rows = connection.execute(
        f"SELECT {', '.join(select_fields)} FROM {_quote(table)} "
        f"WHERE {' AND '.join(where)}", params).fetchall()
    corr_counts = defaultdict(lambda: defaultdict(int))
    for event in device_events:
        corr = event["correlation_id"]
        if corr is not None:
            corr_counts[corr][event["kind"]] += 1
    apis = []
    for row in rows:
        name = _resolve_text(row["name_id"], strings)
        event_class = row["event_class"]
        if event_class == 0:
            source = "CUDA Runtime"
        elif event_class == 1:
            source = "CUDA Driver"
        else:
            source = f"eventClass={event_class}" if event_class is not None else "CUDA API"
        corr = row["correlation_id"]
        apis.append({
            "name": name, "source": source, "start": int(row["start"]),
            "end": int(row["end"]), "tid": row["tid"],
            "correlation_id": corr,
            "device_matches": dict(corr_counts.get(corr, {})) if corr is not None else {},
        })
    return apis, None


def analyze_trace(path, mode):
    connection, resolved_path = _connect_readonly(path)
    try:
        tables = _tables(connection)
        strings = _string_ids(connection, tables)
        nvtx = _load_nvtx(connection, tables, strings)
        candidates = [event for event in nvtx if STEP_LABEL in event["text"]]
        if len(candidates) != 1:
            raise RuntimeError(
                f"{resolved_path}: expected one {STEP_LABEL!r} NVTX range, found {len(candidates)}")
        window = candidates[0]
        if window["end"] <= window["start"]:
            raise RuntimeError(f"{resolved_path}: decode NVTX window has non-positive duration")
        window_start, window_end = window["start"], window["end"]
        stages = []
        for event in nvtx:
            if window["tid"] is not None and event["tid"] != window["tid"]:
                continue
            name = _stage_name(event["text"])
            if name is None:
                continue
            start, end = max(event["start"], window_start), min(event["end"], window_end)
            if end > start:
                stages.append({"name": name, "text": event["text"], "start": start,
                               "end": end, "tid": event["tid"]})

        activities, missing_tables = _device_events(
            connection, tables, window_start, window_end)
        observed_pids = {event["global_pid"] for event in activities
                         if event["global_pid"] is not None}
        null_pid_events = sum(event["global_pid"] is None for event in activities)
        if len(observed_pids) != 1 or null_pid_events:
            raise RuntimeError(
                f"{resolved_path}: expected one non-null device globalPid for "
                f"single-process coverage; found pids={sorted(observed_pids)!r}, "
                f"null_globalPid_events={null_pid_events}")
        process_global_pid = next(iter(observed_pids))
        activities = [event for event in activities
                      if event["global_pid"] == process_global_pid]
        kernel_ranges = [(e["clip_start"], e["clip_end"]) for e in activities
                         if e["kind"] == "kernel"]
        copy_ranges = [(e["clip_start"], e["clip_end"]) for e in activities
                       if e["kind"] in ("memcpy", "memset")]
        device_ranges = [(e["clip_start"], e["clip_end"]) for e in activities]
        kernel_union = union_duration(kernel_ranges)
        copy_union = union_duration(copy_ranges)
        device_union = union_duration(device_ranges)
        window_ns = window_end - window_start

        stage_summary = {}
        for name in STAGE_NAMES:
            selected = [event for event in stages if event["name"] == name]
            layer_counts = defaultdict(int)
            for event in selected:
                found = re.search(r"\.layer(\d+)\b", event["text"], re.IGNORECASE)
                if found:
                    layer_counts[int(found.group(1))] += 1
            row = {
                "range_count": len(selected),
                "nvtx_duration_sum_ms": sum(e["end"] - e["start"] for e in selected) / NS_PER_MS,
            }
            if name == "cpu_gather":
                expected = 2 if mode == "pipeline" else 1
                row.update({
                    "expected_ranges_per_layer": expected,
                    "layers_observed": len(layer_counts),
                    "ranges_per_layer": {str(k): layer_counts[k] for k in sorted(layer_counts)},
                    "layers_not_at_expected_count": [
                        k for k in sorted(layer_counts) if layer_counts[k] != expected],
                })
            stage_summary[name] = row

        copy_enum = _enum_values(connection, tables, "ENUM_CUDA_MEMCPY_OPER")
        h2d_events = []
        copy_kinds = defaultdict(int)
        for event in activities:
            if event["kind"] != "memcpy":
                continue
            label = _resolve_copy_kind(event["copy_kind"], copy_enum)
            copy_kinds[label] += 1
            if not _is_h2d(event["copy_kind"], label):
                continue
            corr = event["correlation_id"]
            h2d_events.append({
                "start_ns": event["clip_start"], "end_ns": event["clip_end"],
                "duration_ms": (event["clip_end"] - event["clip_start"]) / NS_PER_MS,
                "bytes": int(event["bytes"] or 0), "copy_kind": label,
                "correlation_id": corr,
            })

        apis, api_note = _load_cuda_apis(connection, tables, strings, window, activities)
        api_groups = {}
        api_names_by_correlation = defaultdict(set)
        for api in apis:
            key = (api["source"], api["name"])
            group = api_groups.setdefault(key, {
                "source": api["source"], "name": api["name"], "calls": 0,
                "host_duration_ms_sum": 0.0, "correlated_device_activities": defaultdict(int),
            })
            group["calls"] += 1
            group["host_duration_ms_sum"] += (api["end"] - api["start"]) / NS_PER_MS
            for kind, count in api["device_matches"].items():
                group["correlated_device_activities"][kind] += count
            if api["correlation_id"] is not None:
                api_names_by_correlation[api["correlation_id"]].add(api["name"])
        api_summary = []
        for key in sorted(api_groups):
            group = api_groups[key]
            group["correlated_device_activities"] = dict(group["correlated_device_activities"])
            group["host_duration_ms_sum"] = round(group["host_duration_ms_sum"], 6)
            api_summary.append(group)
        for event in h2d_events:
            event["api_names_on_window_thread"] = sorted(
                api_names_by_correlation.get(event["correlation_id"], set()))

        selector_ranges = [event for event in stages if event["name"] == "selector"]
        selector_d2h_ranges = [event for event in stages
                               if event["name"] == "selector_id_d2h"]
        selector_api_groups = {
            "pre_id": {"calls": 0, "host_duration_ms_sum": 0.0, "by_name": {}},
            "id_d2h": {"calls": 0, "host_duration_ms_sum": 0.0, "by_name": {}},
        }
        for api in apis:
            contained_by_selector = any(
                api["tid"] == region["tid"] and api["start"] >= region["start"]
                and api["end"] <= region["end"] for region in selector_ranges)
            if not contained_by_selector:
                continue
            contained_by_d2h = any(
                api["tid"] == region["tid"] and api["start"] >= region["start"]
                and api["end"] <= region["end"] for region in selector_d2h_ranges)
            partition = "id_d2h" if contained_by_d2h else "pre_id"
            group = selector_api_groups[partition]
            duration_ms = (api["end"] - api["start"]) / NS_PER_MS
            group["calls"] += 1
            group["host_duration_ms_sum"] += duration_ms
            name_group = group["by_name"].setdefault(api["name"], {
                "calls": 0, "host_duration_ms_sum": 0.0,
            })
            name_group["calls"] += 1
            name_group["host_duration_ms_sum"] += duration_ms
        for group in selector_api_groups.values():
            group["host_duration_ms_sum"] = round(group["host_duration_ms_sum"], 6)
            group["by_name"] = {
                name: {
                    "calls": value["calls"],
                    "host_duration_ms_sum": round(value["host_duration_ms_sum"], 6),
                }
                for name, value in sorted(group["by_name"].items())
            }

        merged_device = merge_intervals(device_ranges)
        gaps = []
        cursor = window_start
        for start, end in merged_device:
            if start > cursor:
                gaps.append((cursor, start))
            cursor = max(cursor, end)
        if cursor < window_end:
            gaps.append((cursor, window_end))
        largest_gaps = []
        for start, end in sorted(gaps, key=lambda gap: gap[1] - gap[0], reverse=True)[:5]:
            labels = sorted({event["text"] for event in stages
                             if event["start"] < end and event["end"] > start})
            largest_gaps.append({
                "start_ns": start, "end_ns": end,
                "duration_ms": (end - start) / NS_PER_MS,
                "intersecting_host_nvtx_labels": labels,
            })

        h2d_ranges = [(event["start_ns"], event["end_ns"]) for event in h2d_events]
        return {
            "mode": mode,
            "source": str(resolved_path),
            "decode_window": {
                "label": window["text"], "start_ns": window_start,
                "end_ns": window_end, "duration_ms": window_ns / NS_PER_MS,
                "globalTid": window["tid"],
            },
            "process_device_activity_coverage": {
                "ratio": device_union / window_ns,
                "percent": 100.0 * device_union / window_ns,
                "device_activity_union_ms": device_union / NS_PER_MS,
                "uncovered_window_ms": (window_ns - device_union) / NS_PER_MS,
                "kernel_union_ms": kernel_union / NS_PER_MS,
                "copy_union_ms": copy_union / NS_PER_MS,
                "activity_event_counts": {
                    kind: sum(e["kind"] == kind for e in activities)
                    for kind in ("kernel", "memcpy", "memset")
                },
                "single_process_globalPid": process_global_pid,
                "globalPid_count": len(observed_pids),
                "null_globalPid_event_count": null_pid_events,
                "missing_activity_tables": missing_tables,
            },
            "host_nvtx_stage_sums": stage_summary,
            "copy_kind_event_counts": dict(sorted(copy_kinds.items())),
            "h2d": {
                "interval_count": len(h2d_events),
                "bytes": sum(event["bytes"] for event in h2d_events),
                "device_duration_sum_ms": sum(event["duration_ms"] for event in h2d_events),
                "device_interval_union_ms": union_duration(h2d_ranges) / NS_PER_MS,
                "intervals": h2d_events,
            },
            "cuda_api_on_decode_window_thread": {
                "api_call_count": len(apis),
                "host_duration_sum_ms": sum(
                    (api["end"] - api["start"]) / NS_PER_MS for api in apis),
                "by_source_and_name": api_summary,
                "note": api_note,
            },
            "selector_cuda_api_partition": selector_api_groups,
            "largest_device_activity_gaps": largest_gaps,
        }
    finally:
        connection.close()


def _pair_deltas(serial, pipeline):
    def value(trace, path):
        current = trace
        for item in path:
            current = current[item]
        return current
    fields = {
        "decode_window_ms": ("decode_window", "duration_ms"),
        "kernel_union_ms": ("process_device_activity_coverage", "kernel_union_ms"),
        "copy_union_ms": ("process_device_activity_coverage", "copy_union_ms"),
        "device_activity_union_ms": ("process_device_activity_coverage", "device_activity_union_ms"),
        "uncovered_window_ms": ("process_device_activity_coverage", "uncovered_window_ms"),
        "activity_coverage_percentage_points": ("process_device_activity_coverage", "percent"),
        "h2d_bytes": ("h2d", "bytes"),
        "h2d_device_duration_sum_ms": ("h2d", "device_duration_sum_ms"),
    }
    deltas = {name: value(pipeline, path) - value(serial, path)
              for name, path in fields.items()}
    stage_names = sorted(serial["host_nvtx_stage_sums"])
    deltas["host_nvtx_stage_sum_ms_by_name"] = {
        name: (pipeline["host_nvtx_stage_sums"][name]["nvtx_duration_sum_ms"]
               - serial["host_nvtx_stage_sums"][name]["nvtx_duration_sum_ms"])
        for name in stage_names
    }
    return deltas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", type=Path, required=True)
    parser.add_argument("--pipeline", "--candidate", dest="pipeline", type=Path, required=True)
    parser.add_argument("--candidate-kind", choices=("pipeline", "static-mask"),
                        default="pipeline")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidate_key = args.candidate_kind.replace("-", "_")
    delta_key = f"{candidate_key}_minus_serial_diagnostic_deltas"
    result = {
        "serial": analyze_trace(args.serial, "serial"),
        candidate_key: analyze_trace(args.pipeline, args.candidate_kind),
        delta_key: None,
        "limits": [
            "Nsight timings and deltas are diagnostic and must not be presented as speedup or TPOT.",
            "Activity coverage is CUPTI process activity inside the host NVTX window, not full-GPU hardware utilization.",
            "NVTX stage sums can overlap or nest; do not add them into a wall-clock total.",
            "Device gaps and intersecting host NVTX labels are temporal correlation only, not causal attribution.",
            "H2D bytes count complete transfer payloads for activities intersecting the decode window.",
        ],
    }
    result[delta_key] = _pair_deltas(result["serial"], result[candidate_key])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    compact = {
        key: {
            "window_ms": trace["decode_window"]["duration_ms"],
            "kernel_union_ms": trace["process_device_activity_coverage"]["kernel_union_ms"],
            "copy_union_ms": trace["process_device_activity_coverage"]["copy_union_ms"],
            "device_union_ms": trace["process_device_activity_coverage"]["device_activity_union_ms"],
            "uncovered_ms": trace["process_device_activity_coverage"]["uncovered_window_ms"],
            "process_activity_coverage_percent": trace["process_device_activity_coverage"]["percent"],
            "h2d_bytes": trace["h2d"]["bytes"],
        }
        for key, trace in (("serial", result["serial"]), (candidate_key, result[candidate_key]))
    }
    print(json.dumps({"traces": compact,
                      delta_key: result[delta_key],
                      "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
