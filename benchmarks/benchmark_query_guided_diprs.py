"""M10 real-model benchmark: approximate block retrieval vs exact oracle.

Compares, on one real Qwen3-0.6B layer trace (post-RoPE tensors, q_last held
out):

  1. exact full-token Block-DIPR oracle (offline reference only);
  2. flat mean-key representatives;
  3. flat r=4 real representatives;
  4. r=4 K-to-K KNN graph DIPRS;
  5. r=4 sampled-query-guided projected graph DIPRS;
  6. the GPU-resident PyTorch dense-attention baseline (dense_decode_attention;
     NOT FlashAttention -- FlashAttention is only the traced real-engine
     numerical reference).

The timed approximate path never scans all keys and only transfers packed
selected K/V to CUDA.  Index-build time and bytes are reported separately and
are NOT folded into per-query latency.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import torch
from transformers import AutoConfig

from nanovllm import LLM, SamplingParams
from nanovllm.sparse.block_sparse import (
    attention_mass_recovery,
    critical_token_recall,
    dense_decode_attention,
    exact_block_scores,
    gqa_token_scores,
    kv_head_for_query,
    select_dipr_blocks,
    sparse_decode_attention,
    union_block_mask,
)
from nanovllm.sparse.cpu_offload import CPULayerKVStore, route_a_selective
from nanovllm.sparse.representatives import (
    BlockRepresentatives,
    flat_mean_select,
    flat_real_select,
)
from nanovllm.sparse.graph_diprs import (
    BlockGraphIndex,
    refine_candidates,
    union_per_head,
)


def parse_args():
    p = argparse.ArgumentParser(description="M10 query-guided block DIPRS benchmark")
    p.add_argument("--model", default="/opt/models/Qwen3-0.6B")
    p.add_argument("--prompt-length", type=int, default=8192)
    p.add_argument("--fallback-length", type=int, default=4096)
    p.add_argument("--layer", type=int, default=-1)
    p.add_argument("--retrieval-block-size", type=int, default=64)
    p.add_argument("--r", type=int, default=4)
    p.add_argument("--query-samples", type=int, default=128)
    p.add_argument("--recent-tokens", type=int, default=128)
    p.add_argument("--first-tokens", type=int, default=0)
    p.add_argument("--betas", default="16,24,32,48,64,80")
    p.add_argument("--degrees", default="16,32")
    p.add_argument("--max-scored", default="32,64")
    p.add_argument("--l0", type=int, default=16)
    p.add_argument("--projection-topk", type=int, default=8)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--no-pinned", action="store_true")
    p.add_argument("--dense-tol", type=float, default=0.15)
    p.add_argument("--output", default="benchmarks/results/m10_query_guided_diprs.json")
    return p.parse_args()


def deterministic_prompt(length, vocab_size):
    hi = vocab_size - 1000
    return [1000 + (i * 49297 + 17) % max(hi - 1000, 1) for i in range(length)]


def check(cond, msg):
    if not cond:
        raise RuntimeError(f"self-check failed: {msg}")


def sync(d):
    if d.type == "cuda":
        torch.cuda.synchronize()


def med(xs):
    return statistics.median(xs)

def median_timings(runs, warmup):
    reps = runs[warmup:]
    keys = reps[0].timings_ms.keys()
    return {k: statistics.median([r.timings_ms[k] for r in reps]) for k in keys}


def run_one(sel, store, q_cpu, device, rbs, first, recent, scale):
    return route_a_selective(store, q_cpu, sel, device, retrieval_block_size=rbs,
                             first_tokens=first, recent_tokens=recent, scale=scale)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = AutoConfig.from_pretrained(args.model)
    num_layers = cfg.num_hidden_layers
    layer = num_layers // 2 if args.layer < 0 else args.layer
    betas = [float(b) for b in args.betas.split(",") if b.strip()]
    degrees = [int(d) for d in args.degrees.split(",") if d.strip()]
    max_scored_list = [int(m) for m in args.max_scored.split(",") if m.strip()]
    pinned = device.type == "cuda" and not args.no_pinned
    rbs = args.retrieval_block_size

    length = args.prompt_length
    length_note = None
    prompt = None
    while True:
        llm = LLM(args.model, enforce_eager=True, tensor_parallel_size=1,
                  max_num_seqs=1, max_model_len=length + 512,
                  enable_reusable_cache=False, enable_cpu_cache=False,
                  enable_cache_metrics=False, gpu_memory_utilization=0.85)
        try:
            prompt = deterministic_prompt(length, cfg.vocab_size)
            llm.arm_attention_trace(layer, query_samples=args.query_samples)
            llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=1,
                                                  ignore_eos=True), use_tqdm=False)
            trace = llm.retrieve_attention_trace()
            check(trace is not None, "trace not captured")
            break
        except RuntimeError as e:
            llm.exit()
            if length <= args.fallback_length:
                raise
            length_note = f"prompt {args.prompt_length} failed ({str(e)[:120]}); using {args.fallback_length}"
            length = args.fallback_length
            continue

    try:
        check(not trace.q_last.is_cuda and not trace.k.is_cuda, "trace must be CPU")
        check(trace.q_samples is not None, "q_samples not captured")
        T = trace.num_tokens
        q_last = trace.q_last
        k = trace.k
        v = trace.v
        o_last = trace.o_last
        q_samples = trace.q_samples
        q_pos = trace.q_sample_positions.tolist()
        Hq, D = q_last.shape
        Hkv = k.shape[1]
        scale = D ** -0.5
        check(T == length, f"trace tokens {T} != prompt {length}")
        # q_last held out: sampled positions must exclude the final prompt position.
        check(all(0 <= p < T - 1 for p in q_pos), "q_sample positions include q_last")
        check(q_pos == sorted(set(q_pos)), "q_sample positions not unique/sorted")

        # dense PyTorch replay vs traced FlashAttention (real-engine reference)
        q_g = q_last.to(device)
        kf = k.to(device); vf = v.to(device)
        dense_gpu = dense_decode_attention(q_g, kf, vf, scale=scale)
        diff = dense_gpu - o_last.to(device)
        dense_vs_flash = {
            "max_abs_error": float(diff.abs().max().item()),
            "rel_l2_error": float((diff.norm() / o_last.to(device).norm().clamp_min(1e-12)).item()),
        }
        check(dense_vs_flash["rel_l2_error"] <= args.dense_tol,
              f"dense rel_l2={dense_vs_flash['rel_l2_error']:.4f}")

        store = CPULayerKVStore(k, v, pinned=pinned)
        k_f32 = k.float()
        q_f32 = q_last.float()

        # --- exact oracle (offline, outside all approximate timing) --------
        ts_exact = gqa_token_scores(q_f32, k_f32)            # [Hq,T]
        bs_exact = exact_block_scores(ts_exact, rbs)        # [Hq,B]
        dense_probs = torch.softmax(ts_exact * scale, dim=-1)
        g_of = kv_head_for_query(Hq, Hkv)

        reps = BlockRepresentatives(k_f32, rbs, r=args.r)
        B = reps.num_blocks

        # index build (offline; excluded from per-query latency)
        build_records = {}
        graphs = {}
        for deg in degrees:
            gi_knn = BlockGraphIndex(reps)
            gi_knn.build_knn(degree=deg)
            build_records[f"knn_deg{deg}"] = gi_knn.info.__dict__
            gi_qg = BlockGraphIndex(reps)
            gi_qg.build_knn(degree=deg)
            gi_qg._prior_knn_ms = gi_qg.info.build_ms  # its own KNN pre-build
            gi_qg.build_query_guided(q_samples, Hq, degree=deg,
                                     projection_topk=args.projection_topk)
            build_records[f"qg_deg{deg}"] = gi_qg.info.__dict__
            graphs[deg] = (gi_knn, gi_qg)

        # dense GPU baseline timing (outside approximate path)
        sync(device); dt = []
        for _ in range(args.warmup + args.repeats):
            sync(device); t0 = perf_counter()
            dense_decode_attention(q_g, kf, vf, scale=scale)
            sync(device); dt.append((perf_counter() - t0) * 1000)
        dense_time = med(dt[args.warmup:])

        methods = {}

        def record_method(name, per_head_oracle_mask, approx_prewindow, res, work):
            idx = res.indices
            s = int(idx.numel())
            # retrieval recall/precision computed BEFORE forced recent window
            inter = int((approx_prewindow & per_head_oracle_mask).sum())
            denom_o = int(per_head_oracle_mask.sum())
            recall = inter / max(1, denom_o)
            precision = inter / max(1, int(approx_prewindow.sum()))
            out_f = res.output.float()
            d = out_f - dense_gpu.float()
            max_abs = float(d.abs().max().item())
            rel_l2 = float((d.norm() / dense_gpu.float().norm().clamp_min(1e-12)).item())
            mass = attention_mass_recovery(dense_probs, idx)
            entry = {
                "num_selected_blocks": res.num_selected_blocks,
                "num_blocks_total": B,
                "num_selected_tokens": s,
                "selected_token_ratio": s / T,
                "oracle_block_recall": recall,
                "oracle_block_precision": precision,
                "attention_mass_recovery": mass,
                "max_abs_error": max_abs,
                "rel_l2_error": rel_l2,
                "h2d_bytes": res.h2d_bytes,
                "full_kv_bytes": res.full_kv_bytes,
                "packed_full_byte_ratio": res.active_byte_ratio,
                "timings_ms": res.timings_ms,
                "work": work,
                "visited_block_ratio": work.get("visited_block_ratio", None),
                "refined_block_ratio": work.get("refined_block_ratio", None),
            }
            return entry

        # ---- exact oracle per beta (reference; not timed approximate) -------
        def oracle_quality(beta):
            omask = union_block_mask(select_dipr_blocks(bs_exact, beta))
            from nanovllm.sparse.block_sparse import selected_token_indices
            oidx = selected_token_indices(omask, T, rbs,
                                          first_tokens=args.first_tokens,
                                          recent_tokens=args.recent_tokens)
            ogpu = sparse_decode_attention(q_g, kf, vf, oidx.to(device), scale=scale)
            d = ogpu.float() - dense_gpu.float()
            return {
                "selected_token_ratio": int(oidx.numel()) / T,
                "oracle_block_recall": 1.0, "oracle_block_precision": 1.0,
                "attention_mass_recovery": attention_mass_recovery(dense_probs, oidx),
                "max_abs_error": float(d.abs().max().item()),
                "rel_l2_error": float((d.norm() / dense_gpu.float().norm().clamp_min(1e-12)).item()),
                "critical_token_recall": 1.0,
                "beta_raw": beta, "beta_scaled_logit": beta / math.sqrt(D),
            }

        # ---- flat selectors ----------------------------------------------
        def flat_sel(kind, beta):
            def sel():
                res = (flat_mean_select(q_f32, reps, beta) if kind == "mean"
                       else flat_real_select(q_f32, reps, beta))
                pre = res.per_head_mask.any(0)
                # flat selectors scan EVERY block's representatives (all Hq x B).
                work = {
                    "rep_dot_products": res.rep_dot_products,
                    "refined_block_count": res.refined_block_count,
                    "refined_token_dots": res.refined_token_dots,
                    "scanned_block_head_pairs": Hq * B,
                    "scanned_block_ratio": 1.0,
                    "refined_block_head_pairs": res.refined_block_count,
                    "refined_block_ratio": res.refined_block_count / (Hq * B),
                }
                return {"union_mask": pre, "search_ms": res.rep_scan_ms,
                        "refine_ms": res.refine_ms, "work": work, "prewindow": pre}
            return sel

        for beta in betas:
            oracle_mask = union_block_mask(select_dipr_blocks(bs_exact, beta))
            methods[f"exact_block_dipr_oracle_b{beta:g}"] = oracle_quality(beta)
            beta_scaled = beta / math.sqrt(D)
            for kind in ("mean", "real"):
                sel = flat_sel(kind, beta)
                runs = [run_one(sel, store, q_last, device, rbs, args.first_tokens,
                                args.recent_tokens, scale) for _ in range(args.warmup + args.repeats)]
                r = runs[-1]
                rec = record_method(f"flat_{kind}_b{beta:g}", oracle_mask,
                                    r.union_blocks, r, r.work)
                rec["timings_ms"] = median_timings(runs, args.warmup)
                rec["beta_raw"] = beta
                rec["beta_scaled_logit"] = beta_scaled
                rec["critical_token_recall"] = critical_token_recall(ts_exact, beta, r.indices)
                methods[f"flat_{kind}_b{beta:g}"] = rec

            # ---- graph methods -------------------------------------------
            for deg in degrees:
                gi_knn, gi_qg = graphs[deg]
                for maxs in max_scored_list:
                    for gname, gi in (("knn", gi_knn), ("qg", gi_qg)):
                        def sel(g=gi, maxs=maxs, beta=beta):
                            per_final = []
                            t_s = t_r = 0.0
                            rep_dots = 0
                            refined_tokens = 0
                            scored_union = set()
                            refined_union = set()
                            scored_pairs = 0
                            refined_pairs = 0
                            any_trunc = False
                            for h in range(Hq):
                                gv = int(g_of[h])
                                a = perf_counter()
                                tr = g.traverse(q_f32[h], gv, beta, args.l0, maxs)
                                b = perf_counter()
                                fin, rd, _be = refine_candidates(
                                    reps, q_f32[h], gv, tr.rep_candidate_blocks, beta)
                                c = perf_counter()
                                t_s += (b - a) * 1000.0
                                t_r += (c - b) * 1000.0
                                rep_dots += tr.rep_dot_products
                                refined_tokens += rd
                                scored_union.update(tr.scored_blocks)
                                refined_union.update(tr.rep_candidate_blocks)
                                scored_pairs += len(tr.scored_blocks)
                                refined_pairs += len(tr.rep_candidate_blocks)
                                any_trunc = any_trunc or tr.truncated
                                per_final.append(fin)
                            pre = union_per_head(per_final, B)
                            work = {
                                "rep_dot_products": rep_dots,
                                "refined_token_dots": refined_tokens,
                                "scored_block_head_pairs": scored_pairs,
                                "refined_block_head_pairs": refined_pairs,
                                "visited_blocks": len(scored_union),
                                "visited_block_ratio": len(scored_union) / B,
                                "refined_blocks": len(refined_union),
                                "refined_block_ratio": refined_pairs / (Hq * B),
                                "truncated": any_trunc,
                            }
                            return {"union_mask": pre, "search_ms": t_s,
                                    "refine_ms": t_r, "work": work, "prewindow": pre}

                        runs = [run_one(sel, store, q_last, device, rbs,
                                        args.first_tokens, args.recent_tokens, scale)
                                for _ in range(args.warmup + args.repeats)]
                        r = runs[-1]
                        rec = record_method(f"{gname}_deg{deg}_ms{maxs}_b{beta:g}",
                                            oracle_mask, r.union_blocks, r, r.work)
                        rec["timings_ms"] = median_timings(runs, args.warmup)
                        rec["beta_raw"] = beta
                        rec["beta_scaled_logit"] = beta_scaled
                        rec["degree"] = deg
                        rec["max_scored_blocks"] = maxs
                        rec["critical_token_recall"] = critical_token_recall(
                            ts_exact, beta, r.indices)
                        methods[f"{gname}_deg{deg}_ms{maxs}_b{beta:g}"] = rec

        methods["gpu_resident_pytorch_dense_attention"] = {
            "note": "PyTorch dense_decode_attention, full K/V GPU-resident. NOT "
                    "FlashAttention (FlashAttention is only the traced reference).",
            "timings_ms": {"gpu_full_attention": {"median_ms": dense_time}},
            "num_selected_tokens": T, "selected_token_ratio": 1.0,
            "attention_mass_recovery": 1.0, "oracle_block_recall": 1.0,
            "oracle_block_precision": 1.0, "critical_token_recall": 1.0,
            "max_abs_error": 0.0, "rel_l2_error": 0.0, "h2d_bytes": store.resident_bytes,
            "full_kv_bytes": store.resident_bytes, "packed_full_byte_ratio": 1.0,
        }

        reserved_end = torch.cuda.memory_reserved() if device.type == "cuda" else 0
    finally:
        llm.exit()

    result = {
        "milestone": "M10",
        "note": "Offline per-context lab: sampled prefill queries and indexed keys "
                "come from the SAME prompt; q_last is held out. No cross-document "
                "generalization. raw inner product is the authoritative retrieval "
                "score; beta_scaled_logit is a reporting unit only.",
        "command": "benchmarks/benchmark_query_guided_diprs.py",
        "git_head_at_benchmark_start": __import__("subprocess").check_output(
            ["git", "rev-parse", "HEAD"], cwd="/opt/nano-vllm").decode().strip(),
        "git_worktree_dirty_at_benchmark_start": __import__("subprocess").check_output(
            ["git", "status", "--porcelain"], cwd="/opt/nano-vllm").decode().strip() != "",
        "config": {"model": args.model, "prompt_length": length,
                   "length_note": length_note, "layer": layer,
                   "num_hidden_layers": num_layers, "retrieval_block_size": rbs,
                   "r": args.r, "query_samples": args.query_samples,
                   "recent_tokens": args.recent_tokens, "betas": betas,
                   "degrees": degrees, "max_scored": max_scored_list, "l0": args.l0,
                   "projection_topk": args.projection_topk, "warmup": args.warmup,
                   "repeats": args.repeats, "pinned": pinned},
        "head_layout": {"Hq": Hq, "Hkv": Hkv, "D": D, "dtype": str(k.dtype)},
        "trace": {"num_tokens": T, "num_samples": len(q_pos),
                  "first_positions": q_pos[:8], "last_position": q_pos[-1] if q_pos else None},
        "dense_replay_vs_flash_attention": dense_vs_flash,
        "index_build": build_records,
        "graph_degree": {"degree_sweep": degrees, "max_scored_sweep": max_scored_list},
        "bytes": {"full_kv_bytes": store.resident_bytes, "pinned": pinned,
                  "engine_cuda_memory_reserved_end_bytes": reserved_end},
        "methods": methods,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2))
    tmp.replace(out)
    print(f"Wrote {out}")
    print(f"dense vs FlashAttention rel_l2={dense_vs_flash['rel_l2_error']:.4e}")
    for name, m in methods.items():
        if "timings_ms" in m:
            t = m["timings_ms"]
            tot = t.get("total_replay_ms", {}).get("median_ms") if isinstance(t.get("total_replay_ms"), dict) else t.get("gpu_full_attention", {}).get("median_ms")
            print(f"{name:28s} ratio={m.get('selected_token_ratio',0):.3f} "
                  f"recall={m.get('oracle_block_recall',0):.3f} rel={m.get('rel_l2_error',0):.2e}")


if __name__ == "__main__":
    main()
