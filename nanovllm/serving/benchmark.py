"""Measured HTTP benchmark. No simulated timings or SSE-event token counts."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def percentile(values, fraction):
    if not values:
        return None
    xs = sorted(values)
    pos = (len(xs) - 1) * fraction
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo) if lo != hi else xs[lo]


def read_events(response, deadline=None):
    """Parse SSE framing, including comments and multi-line data events."""
    data = []
    for raw in response:
        if deadline is not None and time.perf_counter() > deadline:
            raise TimeoutError("request wall deadline exceeded")
        line = raw.decode("utf-8").rstrip("\r\n")
        if line == "":
            if data:
                yield "\n".join(data)
                data = []
        elif line.startswith("data:"):
            data.append(line[5:].lstrip(" "))
    if data:
        yield "\n".join(data)


def measure_one(endpoint, model, request, timeout_s=300, api_key=None, scheduled=None):
    now = time.perf_counter()
    row = {"id": request["id"], "success": False, "error": None,
           "client_admission_delay_ms": max(0, now - scheduled) * 1000 if scheduled is not None else 0,
           "ttft_ms": None, "tpot_ms": None, "e2e_ms": None,
           "output_tokens": None, "prompt_tokens": None, "token_count_source": "unknown",
           "stream_event_gaps_ms": [], "content_events": 0}
    payload = {"model": model, "messages": request["messages"],
               "max_tokens": request.get("max_tokens", 64), "temperature": request.get("temperature", 0.0),
               "stream": True, "stream_options": {"include_usage": True}}
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    req = urllib.request.Request(endpoint.rstrip("/") + "/v1/chat/completions",
                                 data=json.dumps(payload, allow_nan=False).encode(), headers=headers)
    start = time.perf_counter()
    first = last = None
    id_count = 0
    saw_ids = False
    usage = None
    done = False
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as response:
            if "text/event-stream" not in response.headers.get("Content-Type", ""):
                raise ValueError("backend did not return SSE")
            for data in read_events(response, start + timeout_s):
                if time.perf_counter() - start > timeout_s:
                    raise TimeoutError("request wall deadline exceeded")
                if data == "[DONE]":
                    done = True
                    break
                event = json.loads(data)
                if not isinstance(event, dict):
                    raise ValueError("SSE event must be a JSON object")
                if event.get("error"):
                    raise ValueError(str(event["error"]))
                if event.get("usage"):
                    usage = event["usage"]
                choices = event.get("choices", [])
                if not isinstance(choices, list) or (choices and not isinstance(choices[0], dict)):
                    raise ValueError("invalid choices in SSE event")
                delta = choices[0].get("delta", {}) if choices else {}
                if not isinstance(delta, dict):
                    raise ValueError("invalid delta in SSE event")
                ids = delta.get("token_ids")
                if ids is not None:
                    if not isinstance(ids, list) or any(isinstance(t, bool) or not isinstance(t, int) or t < 0 for t in ids):
                        raise ValueError("invalid token_ids in SSE delta")
                    saw_ids = True
                    id_count += len(ids)
                # Role/usage/finish/heartbeat events are not first tokens.
                if delta.get("content") or delta.get("reasoning_content") or ids:
                    arrived = time.perf_counter()
                    if first is None:
                        first = arrived
                    if last is not None:
                        row["stream_event_gaps_ms"].append((arrived - last) * 1000)
                    last = arrived
                    row["content_events"] += 1
        if not done:
            raise ValueError("stream ended before [DONE]")
        if first is None:
            raise ValueError("stream contained no generated tokens/content")
        if usage is not None and "completion_tokens" in usage:
            tokens = usage["completion_tokens"]
            if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 1:
                raise ValueError("invalid completion_tokens in usage")
            if saw_ids and tokens != id_count:
                raise ValueError("usage completion_tokens disagrees with streamed token IDs")
            row.update(output_tokens=tokens, prompt_tokens=usage.get("prompt_tokens"), token_count_source="server_usage")
        elif saw_ids:
            if id_count < 1:
                raise ValueError("stream supplied token_ids but contained no token IDs")
            row.update(output_tokens=id_count, token_count_source="streamed_token_ids")
        row.update(success=True, ttft_ms=(first - start) * 1000, e2e_ms=(last - start) * 1000)
        if row["output_tokens"] is not None and row["output_tokens"] > 1:
            row["tpot_ms"] = (last - first) * 1000 / (row["output_tokens"] - 1)
    except (OSError, ValueError, TypeError, TimeoutError) as exc:
        row["error"] = str(exc)
    row["request_wall_ms"] = (time.perf_counter() - start) * 1000
    return row


def summarize(rows, wall_seconds, ttft_slo_ms=1000, tpot_slo_ms=100):
    if wall_seconds <= 0:
        raise ValueError("benchmark wall time must be positive")
    ok = [r for r in rows if r["success"]]
    token_counts_known = bool(ok) and all(r["output_tokens"] is not None for r in ok)
    qualifying = [r for r in ok if r["ttft_ms"] <= ttft_slo_ms
                  and (r["output_tokens"] == 1 or r["tpot_ms"] is not None and r["tpot_ms"] <= tpot_slo_ms)]
    report = {"requests": len(rows), "success": len(ok), "success_rate": len(ok) / len(rows) if rows else 0,
              "wall_seconds": wall_seconds, "throughput_requests_per_s": len(ok) / wall_seconds,
              "token_counts_complete": token_counts_known,
              "throughput_tokens_per_s": sum(r["output_tokens"] for r in ok) / wall_seconds if token_counts_known else None,
              "slo": {"ttft_ms": ttft_slo_ms, "tpot_ms": tpot_slo_ms,
                      "joint_attainment_all_requests": len(qualifying) / len(rows) if rows else 0},
              "goodput_requests_per_s": len(qualifying) / wall_seconds,
              "goodput_tokens_per_s": sum(r["output_tokens"] for r in qualifying) / wall_seconds if token_counts_known else None}
    for key in ("ttft_ms", "tpot_ms", "e2e_ms", "client_admission_delay_ms"):
        xs = [r[key] for r in ok if r.get(key) is not None]
        report[key] = {"samples": len(xs), "mean": sum(xs) / len(xs) if xs else None,
                       **{f"p{n}": percentile(xs, n / 100) for n in (50, 95, 99)}}
    return report


def run(endpoint, model, requests, concurrency=2, qps=0, timeout_s=300,
        ttft_slo_ms=1000, tpot_slo_ms=100, api_key=None):
    if concurrency < 1 or not math.isfinite(qps) or qps < 0 or not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("invalid concurrency/qps/timeout")
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = []
        for i, request in enumerate(requests):
            scheduled = start + i / qps if qps else None
            if scheduled is not None:
                time.sleep(max(0, scheduled - time.perf_counter()))
            futures.append(pool.submit(measure_one, endpoint, model, request, timeout_s, api_key, scheduled))
        rows = [f.result() for f in as_completed(futures)]
    wall = time.perf_counter() - start
    rows.sort(key=lambda r: r["id"])
    return rows, summarize(rows, wall, ttft_slo_ms, tpot_slo_ms)


def load_requests(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            item = json.loads(line)
            if not isinstance(item, dict) or not isinstance(item.get("messages"), list):
                raise ValueError("every workload row must contain messages")
            item.setdefault("id", f"req-{len(rows):05d}")
            if not isinstance(item["id"], str):
                raise ValueError("workload IDs must be strings")
            rows.append(item)
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("workload must be nonempty and request IDs unique")
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description="Measure a real NanoKV/OpenAI-compatible SSE service")
    p.add_argument("--endpoint", default="http://127.0.0.1:8080")
    p.add_argument("--model", default="Qwen3-4B")
    p.add_argument("--requests", required=True, help="JSONL messages, max_tokens and unique id")
    p.add_argument("--output", required=True, help="new experiment directory; never overwrite a prior run")
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--qps", type=float, default=0, help="0: closed-loop; positive: paced submissions capped by concurrency")
    p.add_argument("--timeout", type=float, default=300)
    p.add_argument("--ttft-slo-ms", type=float, default=1000)
    p.add_argument("--tpot-slo-ms", type=float, default=100)
    p.add_argument("--engine-config", help="record the exact server config; does not reconfigure the server")
    p.add_argument("--warmup", type=int, default=0, help="number of untimed requests using the first workload row")
    args = p.parse_args(argv)
    if args.warmup < 0 or any(not math.isfinite(x) or x <= 0 for x in (args.timeout, args.ttft_slo_ms, args.tpot_slo_ms)):
        p.error("warmup must be nonnegative and deadlines/SLOs positive")
    requests = load_requests(args.requests)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    api_key = os.getenv("NANOKV_API_KEY")
    workload_bytes = Path(args.requests).read_bytes()
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True))
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    metadata = {"mode": "http_measured", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "commit": commit, "dirty_checkout": dirty, "python": platform.python_version(),
                "platform": platform.platform(), "workload_sha256": hashlib.sha256(workload_bytes).hexdigest(),
                "options": vars(args), "engine_config": json.loads(Path(args.engine_config).read_text()) if args.engine_config else None,
                "metric_contract": "TTFT:first token/content; TPOT:(last-first)/(N-1); throughput:whole measured window",
                "stream_gaps_contract": "SSE token/content event gaps, not per-token ITL when events contain multiple tokens",
                "load_contract": "paced client submissions are concurrency-capped; client admission delay is reported",
                "cache_contract": "warmup may populate prefix cache; this command does not reset backend caches"}
    health_headers = {"Authorization": "Bearer " + api_key} if api_key else {}
    try:
        health_req = urllib.request.Request(args.endpoint.rstrip("/") + "/healthz", headers=health_headers)
        with urllib.request.urlopen(health_req, timeout=5) as response:
            metadata["server_snapshot_before_warmup"] = json.load(response)
    except (OSError, ValueError) as exc:
        metadata["server_snapshot_before_warmup"] = {"unavailable": str(exc)}
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    for _ in range(args.warmup):
        row = measure_one(args.endpoint, args.model, requests[0], args.timeout, api_key)
        if not row["success"]:
            raise RuntimeError("warmup failed: " + row["error"])
    rows, summary = run(args.endpoint, args.model, requests, args.concurrency, args.qps, args.timeout,
                        args.ttft_slo_ms, args.tpot_slo_ms, api_key)
    (output / "requests.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["success"] == summary["requests"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
