"""Deterministic benchmark for the M8 exact block-sparse attention laboratory.

Compares dense decode attention, fixed top-k block selection and exact
Block-DIPR.  Exact block scoring is an **oracle**: it scans every key and is
timed as retrieval/selection work, so it is not expected to improve end-to-end
latency.  Selection and sparse-attention time are reported separately.

Runs on CUDA when available and falls back to CPU.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from nanovllm.sparse.block_sparse import (
    critical_token_recall,
    dense_decode_attention,
    evaluate_sparse_result,
    exact_block_scores,
    gqa_token_scores,
    select_dipr_blocks,
    select_topk_blocks,
    selected_token_indices,
    sparse_decode_attention,
    union_block_mask,
)


def parse_args():
    p = argparse.ArgumentParser(description="M8 block-sparse attention benchmark")
    p.add_argument("--context-length", type=int, default=8192)
    p.add_argument("--retrieval-block-size", type=int, default=64)
    p.add_argument("--num-query-heads", type=int, default=16)
    p.add_argument("--num-kv-heads", type=int, default=4)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--beta", type=float, default=4.0)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--first-tokens", type=int, default=0)
    p.add_argument("--recent-tokens", type=int, default=128)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeats", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=str, default="benchmarks/results/m8_block_sparse.json")
    return p.parse_args()


def sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def time_region(fn, warmup: int, repeats: int, device: torch.device) -> list[float]:
    for _ in range(warmup):
        fn()
    sync(device)
    times = []
    for _ in range(repeats):
        sync(device)
        start = time.perf_counter()
        fn()
        sync(device)
        times.append((time.perf_counter() - start) * 1000.0)  # ms
    return times


def summarize(times: list[float]) -> dict:
    return {
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "min_ms": min(times),
        "max_ms": max(times),
    }


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    T = args.context_length
    Hq = args.num_query_heads
    Hkv = args.num_kv_heads
    D = args.head_dim
    rbs = args.retrieval_block_size

    q = torch.randn(Hq, D, device=device, dtype=torch.float32)
    k = torch.randn(T, Hkv, D, device=device, dtype=torch.float32)
    v = torch.randn(T, Hkv, D, device=device, dtype=torch.float32)
    scale = D ** -0.5

    # Dense oracle (computed once, outside the timed loop).
    dense_out = dense_decode_attention(q, k, v, scale=scale)
    token_scores_untimed = gqa_token_scores(q, k)
    dense_probs = torch.softmax(token_scores_untimed * scale, dim=-1)

    # --- dense attention latency -------------------------------------------
    dense_times = time_region(lambda: dense_decode_attention(q, k, v, scale=scale),
                              args.warmup, args.repeats, device)

    results: dict = {
        "milestone": "M8",
        "note": "Exact Block-DIPR is a correctness oracle; its full key scan is "
                "not expected to improve end-to-end latency.",
        "config": {
            "context_length": T,
            "retrieval_block_size": rbs,
            "num_query_heads": Hq,
            "num_kv_heads": Hkv,
            "head_dim": D,
            "beta": args.beta,
            "top_k": args.top_k,
            "first_tokens": args.first_tokens,
            "recent_tokens": args.recent_tokens,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "seed": args.seed,
        },
        "environment": {
            "device": str(device),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "dense_attention_latency_ms": summarize(dense_times),
        "methods": {},
    }

    num_blocks = (T + rbs - 1) // rbs

    def select_for_mask(per_head_mask: torch.Tensor) -> torch.Tensor:
        union = union_block_mask(per_head_mask)
        return selected_token_indices(
            union, num_tokens=T, retrieval_block_size=rbs,
            first_tokens=args.first_tokens, recent_tokens=args.recent_tokens,
        ), union

    def run_method(name: str, build_mask, oracle: bool = False):
        # Selection region: raw scores + block scores + selection + index build.
        def selection():
            ts = gqa_token_scores(q, k)
            bs = exact_block_scores(ts, rbs)
            mask = build_mask(bs)
            idx, union = select_for_mask(mask)
            return idx, union, ts

        sel_times = time_region(selection, args.warmup, args.repeats, device)
        idx, union, ts = selection()  # one untimed call for metrics

        def sparse_attn():
            return sparse_decode_attention(q, k, v, idx, scale=scale)

        sp_times = time_region(sparse_attn, args.warmup, args.repeats, device)
        sparse_out = sparse_attn()

        metrics = evaluate_sparse_result(dense_out, sparse_out, dense_probs, idx)
        sel_block_ids = union.nonzero(as_tuple=False).flatten().tolist()
        n_sel = int(idx.numel())
        entry = {
            "selection_latency_ms": summarize(sel_times),
            "sparse_attention_latency_ms": summarize(sp_times),
            "total_selection_plus_sparse_ms": {
                "mean_ms": summarize(sel_times)["mean_ms"] + summarize(sp_times)["mean_ms"],
            },
            "selected_block_ids": sel_block_ids,
            "num_selected_blocks": len(sel_block_ids),
            "num_blocks_total": num_blocks,
            "num_selected_tokens": n_sel,
            "selected_token_ratio": n_sel / T,
            "attention_mass_recovery": metrics["attention_mass_recovery"],
            "max_abs_error": metrics["max_abs_error"],
            "rel_l2_error": metrics["rel_l2_error"],
            "oracle": oracle,
        }
        if name == "exact_block_dipr":
            entry["critical_token_recall"] = critical_token_recall(
                ts, args.beta, idx
            )
        results["methods"][name] = entry

    # Full attention: no selection; "sparse" path is the dense call.
    full_sel = torch.zeros(1)
    full_idx = torch.arange(T, device=device)
    full_sparse_times = time_region(
        lambda: sparse_decode_attention(q, k, v, full_idx, scale=scale),
        args.warmup, args.repeats, device,
    )
    results["methods"]["full_attention"] = {
        "selection_latency_ms": None,
        "sparse_attention_latency_ms": summarize(full_sparse_times),
        "total_selection_plus_sparse_ms": {
            "mean_ms": summarize(full_sparse_times)["mean_ms"],
        },
        "selected_block_ids": list(range(num_blocks)),
        "num_selected_blocks": num_blocks,
        "num_blocks_total": num_blocks,
        "num_selected_tokens": T,
        "selected_token_ratio": 1.0,
        "attention_mass_recovery": 1.0,
        "critical_token_recall": 1.0,
        "max_abs_error": 0.0,
        "rel_l2_error": 0.0,
        "oracle": False,
    }

    run_method("top_k_blocks", lambda bs: select_topk_blocks(bs, args.top_k))
    run_method("exact_block_dipr", lambda bs: select_dipr_blocks(bs, args.beta), oracle=True)

    results["timestamp"] = datetime.now(timezone.utc).isoformat()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out_path}")
    for name, m in results["methods"].items():
        print(
            f"{name:18s} sel={m['selection_latency_ms'] and round(m['selection_latency_ms']['mean_ms'],3)}ms "
            f"attn={round(m['sparse_attention_latency_ms']['mean_ms'],3)}ms "
            f"tok_ratio={m['selected_token_ratio']:.3f} "
            f"mass={m['attention_mass_recovery']:.4f} "
            f"rel_l2={m['rel_l2_error']:.3e}"
        )


if __name__ == "__main__":
    main()
