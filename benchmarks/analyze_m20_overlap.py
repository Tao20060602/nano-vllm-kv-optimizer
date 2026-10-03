"""Correlate K H2D device events with V CPU gather; diagnostic, not TPOT."""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    connection = sqlite3.connect(args.sqlite.resolve().as_uri() + "?mode=ro", uri=True)
    rows = connection.execute(
        "SELECT start,end,text,globalTid FROM NVTX_EVENTS "
        "WHERE text LIKE 'm12.cpu_gather.layer%' OR text LIKE 'm12.h2d_pack.layer%' "
        "ORDER BY start").fetchall()
    stages = {}
    for start, end, text, tid in rows:
        kind, layer = text.removeprefix("m12.").split(".layer")
        stages.setdefault(int(layer), {}).setdefault(kind, []).append((start, end, tid))
    details = []
    for layer, ranges in sorted(stages.items()):
        if len(ranges.get("cpu_gather", [])) != 2 or len(ranges.get("h2d_pack", [])) != 2:
            raise RuntimeError(f"layer {layer}: expected two gather and two pack ranges")
        vstart, vend, tid = ranges["cpu_gather"][1]
        hstart, hend, htid = ranges["h2d_pack"][0]
        copies = connection.execute(
            "SELECT m.start,m.end,m.bytes,m.copyKind,r.start,r.end "
            "FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
            "JOIN CUPTI_ACTIVITY_KIND_MEMCPY m ON m.correlationId=r.correlationId "
            "JOIN StringIds s ON s.id=r.nameId "
            "WHERE r.globalTid=? AND r.start>=? AND r.end<=? "
            "AND s.value LIKE 'cudaMemcpyAsync%'", (htid, hstart, hend)).fetchall()
        if len(copies) != 1:
            raise RuntimeError(f"layer {layer}: expected one correlated K H2D, got {copies}")
        kstart, kend, nbytes, kind, apistart, apiend = copies[0]
        overlap = max(0, min(kend, vend) - max(kstart, vstart))
        details.append({
            "layer": layer, "k_h2d_bytes": nbytes, "copy_kind": kind,
            "k_device_start_ns": kstart, "k_device_end_ns": kend,
            "v_cpu_start_ns": vstart, "v_cpu_end_ns": vend,
            "k_api_ms": (apiend-apistart)/1e6,
            "k_device_ms": (kend-kstart)/1e6,
            "v_cpu_ms": (vend-vstart)/1e6, "overlap_ms": overlap/1e6,
        })
    if not details:
        raise RuntimeError("no pipeline layer ranges found")
    output = {
        "source": str(args.sqlite), "layers": len(details),
        "layers_with_overlap": sum(row["overlap_ms"] > 0 for row in details),
        "summed_k_device_ms": sum(row["k_device_ms"] for row in details),
        "summed_v_cpu_ms": sum(row["v_cpu_ms"] for row in details),
        "summed_overlap_ms": sum(row["overlap_ms"] for row in details),
        "layers_detail": details,
        "limits": "one profiled step; profiler perturbs times; intersection is not unprofiled speedup",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({key: value for key, value in output.items() if key != "layers_detail"}, indent=2))


if __name__ == "__main__":
    main()
