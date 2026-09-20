"""Generate CSV, JSON, SVG, and conclusions from raw prefix-reuse results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(math.ceil(len(ordered) * fraction) - 1, 0)]


def group_key(record: dict) -> tuple:
    return (
        record["mode"],
        record["point"],
        record["prefix_tokens"],
        record["suffix_tokens"],
        record["prompt_tokens"],
        record["reuse_ratio"],
        record["max_tokens"],
    )


def summarize(raw_files: list[Path]) -> tuple[list[dict], list[dict]]:
    all_records = []
    for path in raw_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for record in payload["records"]:
            record["source"] = str(path)
            all_records.append(record)
    groups = defaultdict(list)
    for record in all_records:
        if not record["warmup"]:
            groups[group_key(record)].append(record)
    summary = []
    for key, records in sorted(groups.items()):
        mode, point, prefix, suffix, prompt, ratio, max_tokens = key
        ttft = [row["metrics"]["ttft_ms"] for row in records]
        wall = [row["wall_time_ms"] for row in records]
        lookup = [row["metrics"]["lookup_time_ms"] for row in records]
        load = [row["metrics"]["load_time_ms"] for row in records]
        prefill = [row["metrics"]["prefill_time_ms"] for row in records]
        summary.append(
            {
                "mode": mode,
                "point": point,
                "prefix_tokens": prefix,
                "suffix_tokens": suffix,
                "prompt_tokens": prompt,
                "reuse_ratio": ratio,
                "max_tokens": max_tokens,
                "measurements": len(records),
                "ttft_p50_ms": statistics.median(ttft),
                "ttft_p95_ms": percentile(ttft, 0.95),
                "wall_p50_ms": statistics.median(wall),
                "wall_p95_ms": percentile(wall, 0.95),
                "lookup_p50_ms": statistics.median(lookup),
                "load_p50_ms": statistics.median(load),
                "prefill_p50_ms": statistics.median(prefill),
                "lookup_share_of_ttft_p50": (
                    statistics.median(lookup) / statistics.median(ttft)
                    if statistics.median(ttft)
                    else 0.0
                ),
                "gpu_peak_allocated_bytes_max": max(
                    row["gpu_peak_allocated_bytes"] for row in records
                ),
                "gpu_peak_reserved_bytes_max": max(
                    row["gpu_peak_reserved_bytes"] for row in records
                ),
                "cpu_resident_bytes_max": max(
                    row["cpu_cache"].get("store", {}).get("resident_bytes", 0)
                    for row in records
                ),
                "h2d_bytes_per_request": statistics.median(
                    row["metrics"]["h2d_bytes"] for row in records
                ),
                "source": records[0]["source"],
            }
        )
    return all_records, summary


def write_svg(summary: list[dict], path: Path) -> None:
    rows = [
        row
        for row in summary
        if row["point"] == "break_even" and row["max_tokens"] == 1
    ]
    modes = ("cold", "gpu", "cpu_pageable", "cpu_pinned")
    colors = {
        "cold": "#d95f02",
        "gpu": "#1b9e77",
        "cpu_pageable": "#7570b3",
        "cpu_pinned": "#e7298a",
    }
    width, height = 900, 520
    left, top, plot_w, plot_h = 80, 40, 760, 390
    values = [row["ttft_p50_ms"] for row in rows]
    max_y = max(values) * 1.1 if values else 1
    prefixes = sorted({row["prefix_tokens"] for row in rows})

    def x(prefix):
        return left + prefixes.index(prefix) * plot_w / max(len(prefixes) - 1, 1)

    def y(value):
        return top + plot_h - value / max_y * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="black"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="black"/>',
        '<text x="450" y="505" text-anchor="middle">Reusable prefix tokens</text>',
        '<text x="18" y="240" transform="rotate(-90 18 240)" text-anchor="middle">TTFT p50 (ms)</text>',
    ]
    for prefix in prefixes:
        parts.append(f'<text x="{x(prefix)}" y="455" text-anchor="middle">{prefix}</text>')
    for mode_index, mode in enumerate(modes):
        mode_rows = sorted(
            (row for row in rows if row["mode"] == mode),
            key=lambda row: row["prefix_tokens"],
        )
        if not mode_rows:
            continue
        points = " ".join(
            f'{x(row["prefix_tokens"])},{y(row["ttft_p50_ms"])}'
            for row in mode_rows
        )
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="{colors[mode]}" stroke-width="3"/>'
        )
        for row in mode_rows:
            parts.append(
                f'<circle cx="{x(row["prefix_tokens"])}" cy="{y(row["ttft_p50_ms"])}" r="4" fill="{colors[mode]}"/>'
            )
        legend_y = 25 + mode_index * 20
        parts.append(f'<rect x="650" y="{legend_y - 10}" width="14" height="4" fill="{colors[mode]}"/>')
        parts.append(f'<text x="670" y="{legend_y}">{mode}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def conclusions(summary: list[dict]) -> dict:
    indexed = {
        (row["mode"], row["prefix_tokens"], row["suffix_tokens"], row["max_tokens"]): row
        for row in summary
        if row["point"] in ("break_even", "full_request_32")
    }
    break_even = {}
    for mode in ("gpu", "cpu_pageable", "cpu_pinned"):
        winners = []
        for prefix in BREAK_EVEN_PREFIXES:
            cold = indexed.get(("cold", prefix, 16, 1))
            candidate = indexed.get((mode, prefix, 16, 1))
            if cold and candidate and candidate["ttft_p50_ms"] < cold["ttft_p50_ms"]:
                winners.append(prefix)
        break_even[mode] = min(winners) if winners else None
    slower = []
    for mode in ("gpu", "cpu_pageable", "cpu_pinned"):
        for prefix in BREAK_EVEN_PREFIXES:
            cold = indexed.get(("cold", prefix, 16, 1))
            candidate = indexed.get((mode, prefix, 16, 1))
            if cold and candidate and candidate["ttft_p50_ms"] >= cold["ttft_p50_ms"]:
                slower.append(
                    {
                        "mode": mode,
                        "prefix_tokens": prefix,
                        "cold_p50_ms": cold["ttft_p50_ms"],
                        "candidate_p50_ms": candidate["ttft_p50_ms"],
                    }
                )
    return {"break_even_prefix_tokens": break_even, "not_faster_than_cold": slower}


BREAK_EVEN_PREFIXES = (256, 512, 1024, 2048, 4096)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    _, summary = summarize(args.inputs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "summary.json"
    csv_path = args.output_dir / "summary.csv"
    svg_path = args.output_dir / "ttft_break_even.svg"
    conclusions_path = args.output_dir / "conclusions.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    write_svg(summary, svg_path)
    conclusions_path.write_text(
        json.dumps(conclusions(summary), indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(summary)} summary rows to {args.output_dir}")


if __name__ == "__main__":
    main()
