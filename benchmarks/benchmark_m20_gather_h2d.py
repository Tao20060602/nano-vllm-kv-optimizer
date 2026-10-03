"""Matched, synchronized gather+H2D microbenchmark; not model TPOT."""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids-log", type=Path,
                        default=Path("bench_logs/m19_decode_raw_ab_3_0.json"))
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.cpu_threads, args.samples, args.rounds) < 1:
        raise SystemExit("threads, samples, and rounds must be positive")
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(20)
    reference = json.loads(args.ids_log.read_text())
    history = reference["payload"]["ids_history_by_layer"]
    B, H, D, K = 64, 8, 128, 32
    nblocks = reference["config"]["seq_len"] // B
    # Traverse successive steps across layers, not one layer's repeated IDs.
    rows = [layer[step] for step in range(min(map(len, history))) for layer in history]
    if any(len(row) != K or min(row) < 0 or max(row) >= nblocks for row in rows):
        raise SystemExit("expected valid fixed Top-32 IDs")
    ids_list = [torch.tensor(row, dtype=torch.long) for row in rows[:args.samples]]
    if not ids_list:
        raise SystemExit("no selected IDs")
    k = torch.randn(nblocks * B, H, D, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    sk = torch.empty(K * B, H, D, dtype=k.dtype, pin_memory=True)
    sv = torch.empty_like(sk, pin_memory=True)
    gk, gv = torch.empty_like(sk, device="cuda"), torch.empty_like(sv, device="cuda")
    pointers = sk.data_ptr(), sv.data_ptr()

    def run(mode, ids):
        idx = (ids[:, None] * B + torch.arange(B)).reshape(-1)
        if mode == "serial":
            torch.index_select(k, 0, idx, out=sk)
            torch.index_select(v, 0, idx, out=sv)
            gk.copy_(sk, non_blocking=True)
            gv.copy_(sv, non_blocking=True)
        elif mode == "kv_overlap":
            torch.index_select(k, 0, idx, out=sk)
            gk.copy_(sk, non_blocking=True)
            torch.index_select(v, 0, idx, out=sv)
            gv.copy_(sv, non_blocking=True)
        else:
            chunk = int(mode.removeprefix("chunk")) * B
            for start in range(0, K * B, chunk):
                end = min(start + chunk, K * B)
                torch.index_select(k, 0, idx[start:end], out=sk[start:end])
                gk[start:end].copy_(sk[start:end], non_blocking=True)
                torch.index_select(v, 0, idx[start:end], out=sv[start:end])
                gv[start:end].copy_(sv[start:end], non_blocking=True)
        # Distinct staging slices are never overwritten within an iteration;
        # completion before the next iteration prevents host-buffer races.
        torch.cuda.synchronize()

    modes = ["serial", "kv_overlap", "chunk8", "chunk16"]
    for mode in modes:
        for ids in ids_list[:4]:
            run(mode, ids)
            idx = (ids[:, None] * B + torch.arange(B)).reshape(-1)
            assert torch.equal(gk.cpu(), k[idx])
            assert torch.equal(gv.cpu(), v[idx])
            assert (sk.data_ptr(), sv.data_ptr()) == pointers
            assert sk.is_pinned() and sv.is_pinned()
    rounds = []
    for rnd in range(args.rounds):
        # Rotate order so every variant occupies each measurement position.
        order = modes[rnd % len(modes):] + modes[:rnd % len(modes)]
        measurements = {}
        for mode in order:
            samples = []
            for ids in ids_list:
                torch.cuda.synchronize()
                start = time.perf_counter()
                run(mode, ids)
                samples.append((time.perf_counter() - start) * 1000)
            measurements[mode] = {"median_ms": statistics.median(samples),
                                  "mean_ms": statistics.mean(samples), "samples_ms": samples}
        rounds.append({"order": order, "measurements": measurements})
    result = {
        "measurement": "CPU gather + H2D completion wall; not attention or TPOT",
        "torch_version": torch.__version__, "threads": torch.get_num_threads(),
        "gpu": torch.cuda.get_device_name(), "source_log": str(args.ids_log),
        "shape": {"nblocks": nblocks, "B": B, "Hkv": H, "D": D, "K": K},
        "payload_mib_per_iteration": 8, "sample_id_sets": len(ids_list),
        "rounds": rounds,
        "summary": {mode: {
            "median_of_round_medians_ms": statistics.median([
                row["measurements"][mode]["median_ms"] for row in rounds]),
            "round_medians_ms": [row["measurements"][mode]["median_ms"] for row in rounds],
        } for mode in modes},
        "correctness": "exact GPU K/V for all variants on four ID sets; pinned pointers stable",
        "limits": "one repeated synthetic source; no attention, model, or serving workload",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
