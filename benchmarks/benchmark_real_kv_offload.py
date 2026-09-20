"""M9 real-model KV trace + synchronous CPU-offload benchmark.

Pipeline:
  1. load local Qwen3-0.6B through nano-vLLM (eager, TP=1, batch 1, caches off);
  2. capture one chosen layer's post-RoPE q_last/k/v and FlashAttention o_last
     from a cold, single-sequence prefill;
  3. validate a dense PyTorch replay against the traced FlashAttention output;
  4. keep that layer's historical K/V in CPU memory (pinned on CUDA);
  5. compare full-GPU attention with exact top-k and exact Block-DIPR Route A
     (beta sweep), reporting search/gather/H2D/attention/total latency,
     selection/quality metrics and active packed-vs-full K/V byte ratios.

The exact CPU scan is an oracle and may be slow.  Active replay-buffer bytes are
reported separately from the engine's process-wide reserved memory; this lab
does NOT claim a reduction in torch.cuda.memory_reserved().
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
from transformers import AutoConfig

from nanovllm import LLM, SamplingParams
from nanovllm.sparse.block_sparse import (
    attention_mass_recovery,
    critical_token_recall,
    dense_decode_attention,
    exact_block_scores,
    gqa_token_scores,
    select_dipr_blocks,
    select_topk_blocks,
)
from nanovllm.sparse.cpu_offload import CPULayerKVStore, route_a_replay


def parse_args():
    p = argparse.ArgumentParser(description="M9 real-model CPU KV offload benchmark")
    p.add_argument("--model", type=str, default="/opt/models/Qwen3-0.6B")
    p.add_argument("--prompt-length", type=int, default=2048)
    p.add_argument("--layer", type=int, default=-1, help="-1 = middle layer")
    p.add_argument("--retrieval-block-size", type=int, default=64)
    p.add_argument("--recent-tokens", type=int, default=128)
    p.add_argument("--first-tokens", type=int, default=0)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--betas", type=str,
                   default="0.5,1,2,4,8,16,24,32,48,64,80",
                   help="raw (unscaled) Block-DIPR beta sweep; Qwen3 RMSNorm "
                        "places the informative transition around beta 16-80")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--no-pinned", action="store_true")
    p.add_argument("--dense-tol", type=float, default=0.15,
                   help="max allowed relative-L2 dense-replay vs FlashAttention error")
    p.add_argument("--output", type=str,
                   default="benchmarks/results/m9_real_kv_offload.json")
    return p.parse_args()


def deterministic_prompt(length: int, vocab_size: int) -> list[int]:
    # Fresh, deterministic, no special-token ids; caches are disabled so there
    # is no paged prefix hit regardless.
    safe_hi = vocab_size - 1000
    return [1000 + (i * 49297 + 17) % max(safe_hi - 1000, 1) for i in range(length)]


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def time_cuda(fn, warmup, repeats, device):
    for _ in range(warmup):
        fn()
    sync(device)
    times = []
    for _ in range(repeats):
        sync(device)
        t0 = time.perf_counter()
        fn()
        sync(device)
        times.append((time.perf_counter() - t0) * 1000.0)
    return times


def summarize(times):
    return {
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "min_ms": min(times),
        "max_ms": max(times),
    }


def summarize_regions(runs, key):
    return summarize([r.timings_ms[key] for r in runs])


def check(cond, msg):
    if not cond:
        raise RuntimeError(f"self-check failed: {msg}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hf = AutoConfig.from_pretrained(args.model)
    num_layers = hf.num_hidden_layers
    layer = num_layers // 2 if args.layer < 0 else args.layer
    betas = [float(b) for b in args.betas.split(",") if b.strip()]
    pinned = device.type == "cuda" and not args.no_pinned
    T = args.prompt_length
    rbs = args.retrieval_block_size

    # --- 1. load engine (eager, TP=1, batch 1, prefix/CPU caches off) ------
    llm = LLM(
        args.model,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_num_seqs=1,
        max_model_len=T + 512,
        enable_reusable_cache=False,
        enable_cpu_cache=False,
        enable_cache_metrics=False,
        gpu_memory_utilization=0.85,
    )
    try:
        prompt = deterministic_prompt(T, hf.vocab_size)

        # --- 2. arm one-shot trace, then run a cold prefill ----------------
        llm.arm_attention_trace(layer)
        llm.generate(
            [prompt],
            SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True),
            use_tqdm=False,
        )
        trace = llm.retrieve_attention_trace()
        check(trace is not None, "attention trace was not captured")
        check(not trace.q_last.is_cuda and not trace.k.is_cuda,
              "trace tensors must be CPU copies")
        llm.clear_attention_trace()
        reserved_after_prefill = (
            torch.cuda.memory_reserved() if device.type == "cuda" else 0
        )

        q_last = trace.q_last          # [Hq, D] CPU bf16
        k = trace.k                    # [T, Hkv, D] CPU bf16
        v = trace.v
        o_last = trace.o_last          # [Hq, D] CPU bf16
        Hq, D = q_last.shape
        Hkv = k.shape[1]
        scale = D ** -0.5
        check(k.shape[0] == T and v.shape[0] == T,
              f"traced tokens {k.shape[0]} != prompt length {T}")

        # --- 3. dense PyTorch replay vs traced FlashAttention --------------
        q_g = q_last.to(device)
        k_full_g = k.to(device)
        v_full_g = v.to(device)
        dense_gpu = dense_decode_attention(q_g, k_full_g, v_full_g, scale=scale)
        diff = dense_gpu - o_last.to(device)
        dense_vs_flash = {
            "max_abs_error": float(diff.abs().max().item()),
            "rel_l2_error": float(
                (diff.norm() / o_last.to(device).norm().clamp_min(1e-12)).item()
            ),
        }
        check(torch.isfinite(dense_gpu.float()).all(), "dense replay non-finite")
        check(dense_vs_flash["rel_l2_error"] <= args.dense_tol,
              f"dense replay rel_l2={dense_vs_flash['rel_l2_error']:.4f} "
              f"exceeds tol={args.dense_tol}")

        # --- 4. CPU store and CPU-f32 oracle probabilities -----------------
        store = CPULayerKVStore(k, v, pinned=pinned)
        ts_cpu = gqa_token_scores(q_last.float(), k.float())   # [Hq,T] f32
        block_scores_cpu = exact_block_scores(ts_cpu, rbs)
        qs = torch.quantile(
            block_scores_cpu.flatten().float(),
            torch.tensor([0.0, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]),
        ).tolist()
        block_score_stats = {
            "min": qs[0], "p50": qs[1], "p75": qs[2], "p90": qs[3],
            "p95": qs[4], "p99": qs[5], "max": qs[6],
            "mean": float(block_scores_cpu.float().mean().item()),
        }
        dense_probs_cpu = torch.softmax(ts_cpu * scale, dim=-1)
        element_size = k.element_size()
        full_kv_bytes = 2 * T * Hkv * D * element_size
        check(store.resident_bytes == full_kv_bytes,
              "CPU store byte accounting mismatch")

        # --- 5. full-GPU attention baseline timing -------------------------
        full_times = time_cuda(
            lambda: dense_decode_attention(q_g, k_full_g, v_full_g, scale=scale),
            args.warmup, args.repeats, device,
        )

        def run_route(build_mask):
            runs = []
            for _ in range(args.warmup + args.repeats):
                runs.append(route_a_replay(
                    store, q_last, build_mask, device,
                    retrieval_block_size=rbs,
                    first_tokens=args.first_tokens,
                    recent_tokens=args.recent_tokens,
                    scale=scale,
                ))
            return runs[args.warmup:]  # drop warmups

        def metrics_from_runs(runs, *, beta=None):
            r = runs[-1]
            idx = r.indices
            s = r.num_selected_tokens
            # self-checks
            check(torch.equal(idx, torch.unique(idx.sort().values)),
                  "selected indices not sorted/unique")
            check(0 < s <= T, f"bad selected token count {s}")
            check(r.h2d_bytes == r.packed_k_bytes + r.packed_v_bytes,
                  "h2d_bytes != packed_k + packed_v")
            check(abs(r.active_byte_ratio - s / T) < 1e-9,
                  "active byte ratio != S/T")
            out_f = r.output.float()
            dense_f = dense_gpu.float()
            d = out_f - dense_f
            max_abs = float(d.abs().max().item())
            rel_l2 = float((d.norm() / dense_f.norm().clamp_min(1e-12)).item())
            check(torch.isfinite(out_f).all(), "sparse output non-finite")
            check(math_isfinite(max_abs, rel_l2), "error metrics non-finite")
            mass = attention_mass_recovery(dense_probs_cpu, idx)
            entry = {
                "timings_ms": {
                    "cpu_search": summarize_regions(runs, "cpu_search_ms"),
                    "cpu_gather": summarize_regions(runs, "cpu_gather_ms"),
                    "h2d": summarize_regions(runs, "h2d_ms"),
                    "gpu_packed_attention": summarize_regions(
                        runs, "gpu_packed_attention_ms"),
                    "total_replay": summarize_regions(runs, "total_replay_ms"),
                },
                "num_selected_blocks": r.num_selected_blocks,
                "num_blocks_total": (T + rbs - 1) // rbs,
                "num_selected_tokens": s,
                "selected_token_ratio": s / T,
                "attention_mass_recovery": mass,
                "max_abs_error": max_abs,
                "rel_l2_error": rel_l2,
                "packed_k_bytes": r.packed_k_bytes,
                "packed_v_bytes": r.packed_v_bytes,
                "h2d_bytes": r.h2d_bytes,
                "full_kv_bytes": r.full_kv_bytes,
                "packed_full_byte_ratio": r.active_byte_ratio,
                "pinned": r.pinned,
            }
            if beta is not None:
                recall = critical_token_recall(ts_cpu, beta, idx)
                entry["critical_token_recall"] = recall
                check(abs(recall - 1.0) < 1e-9,
                      f"Block-DIPR critical recall={recall} != 1.0")
            return entry

        methods = {
            "full_gpu_attention": {
                "timings_ms": {"gpu_full_attention": summarize(full_times)},
                "num_selected_tokens": T,
                "selected_token_ratio": 1.0,
                "attention_mass_recovery": 1.0,
                "critical_token_recall": 1.0,
                "max_abs_error": 0.0,
                "rel_l2_error": 0.0,
                "h2d_bytes": full_kv_bytes,
                "full_kv_bytes": full_kv_bytes,
                "packed_full_byte_ratio": 1.0,
                "note": "baseline with full K/V GPU-resident (not offload)",
            }
        }

        topk_runs = run_route(lambda bs: select_topk_blocks(bs, args.top_k))
        methods[f"top_k_{args.top_k}_route_a"] = metrics_from_runs(topk_runs)

        for beta in betas:
            runs = run_route(lambda bs, beta=beta: select_dipr_blocks(bs, beta))
            methods[f"block_dipr_beta_{beta:g}_route_a"] = metrics_from_runs(
                runs, beta=beta
            )

        reserved_end = torch.cuda.memory_reserved() if device.type == "cuda" else 0
    finally:
        llm.exit()

    result = {
        "milestone": "M9",
        "note": "Exact CPU scan is an oracle. Active packed/full K/V bytes are "
                "per-layer replay-buffer bytes, NOT process memory_reserved "
                "savings; the engine still preallocates its paged KV cache.",
        "config": {
            "model": args.model,
            "prompt_length": T,
            "layer": layer,
            "num_hidden_layers": num_layers,
            "retrieval_block_size": rbs,
            "recent_tokens": args.recent_tokens,
            "first_tokens": args.first_tokens,
            "top_k": args.top_k,
            "betas": betas,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "pinned": pinned,
        },
        "head_layout": {
            "num_query_heads": Hq,
            "num_kv_heads": Hkv,
            "head_dim": D,
            "dtype": str(k.dtype),
        },
        "raw_block_score_stats": block_score_stats,
        "environment": {
            "device": str(device),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "python": platform.python_version(),
        },
        "dense_replay_vs_flash_attention": dense_vs_flash,
        "bytes": {
            "full_kv_bytes": full_kv_bytes,
            "cpu_store_resident_bytes": store.resident_bytes,
            "cpu_store_pinned": store.pinned,
            "engine_cuda_memory_reserved_after_prefill_bytes": reserved_after_prefill,
            "engine_cuda_memory_reserved_end_bytes": reserved_end,
            "note": "memory_reserved is the live engine's process-wide pool and "
                    "is not reduced by this layer laboratory.",
        },
        "methods": methods,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"Wrote {out}")
    print(f"dense vs FlashAttention rel_l2 = "
          f"{dense_vs_flash['rel_l2_error']:.4e} "
          f"(max_abs={dense_vs_flash['max_abs_error']:.4e})")
    for name, m in methods.items():
        ratio = m["packed_full_byte_ratio"]
        mass = m["attention_mass_recovery"]
        rel = m["rel_l2_error"]
        print(f"{name:32s} byte_ratio={ratio:.3f} mass={mass:.4f} rel_l2={rel:.3e}")


def math_isfinite(*vals):
    return all(v == v and abs(v) != float("inf") for v in vals)


if __name__ == "__main__":
    main()
