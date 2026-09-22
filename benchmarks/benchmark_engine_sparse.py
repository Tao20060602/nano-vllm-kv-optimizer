"""M11 end-to-end sparse-generation benchmark (real Qwen3-0.6B, eager, TP=1).

Deterministic exactly-2048-token needle prompt; dense feature-off baseline vs
every sparse selector; JSON + CSV; offline block-size replay sweep.
Writes via atomic temp-file replace.
"""
import argparse
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

import torch

os.environ.setdefault("HF_HOME", "/opt/models/.cache/huggingface")

from nanovllm import LLM, SamplingParams  # noqa: E402

MODEL = "/opt/models/Qwen3-0.6B"
OUT = Path("/opt/nano-vllm/benchmarks/results")
OUT.mkdir(parents=True, exist_ok=True)

SELECTORS = ["full", "exact_dipr", "top_k", "mean", "real", "knn_graph", "query_guided"]
NUM_GEN = 16
BETA_RAW = 48.0
HEAD_DIM = 128


def make_prompt(tokenizer):
    needle = "The secret passkey is 74291."
    filler = ("The solar system contains eight planets orbiting the Sun. "
              "Each follows an elliptical path at a different distance. ")
    q = "What is the secret passkey? Reply with only the number."
    needle_ids = tokenizer.encode(needle + " ")
    filler_ids = tokenizer.encode(filler)
    question_ids = tokenizer.encode(" " + q)
    middle_len = 2048 - len(needle_ids) - len(question_ids)
    assert middle_len > 0
    middle = (filler_ids * ((middle_len // len(filler_ids)) + 1))[:middle_len]
    ids = needle_ids + middle + question_ids
    assert len(ids) == 2048, f"prompt must be exactly 2048, got {len(ids)}"
    assert ids[-len(question_ids):] == question_ids, "question must survive truncation"
    return ids


def timed_run(engine, prompt_ids, sp):
    engine.add_request(prompt_ids, sp)
    t0 = time.perf_counter()
    engine.step()
    ttft = (time.perf_counter() - t0) * 1000.0
    seq = engine.scheduler.running[0] if engine.scheduler.running else None
    decode_ms = []
    while not engine.is_finished():
        t1 = time.perf_counter()
        engine.step()
        decode_ms.append((time.perf_counter() - t1) * 1000.0)
    tok = list(seq.completion_token_ids) if seq is not None else []
    return tok, ttft, decode_ms


def pct(xs, p):
    if not xs:
        return 0.0
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round((p / 100) * (len(s) - 1)))))
    return s[k]


def get_rt(engine, layer_id=14):
    for m in engine.model_runner.model.modules():
        rt = getattr(m, "sparse_rt", None)
        if rt is not None and rt.layer_id == layer_id:
            return rt
    return None


def atomic_write(path, text):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", type=Path,
                        default=OUT / "m11_engine_sparse.json")
    parser.add_argument("--output-csv", type=Path,
                        default=OUT / "m11_engine_sparse.csv")
    args = parser.parse_args()
    max_model_len = 2048 + NUM_GEN + 4
    base_kwargs = dict(model=MODEL, enforce_eager=True, tensor_parallel_size=1,
                       max_num_seqs=1, max_model_len=max_model_len)

    # dense feature-off baseline
    dense = LLM(**base_kwargs)
    tok_ids = make_prompt(dense.tokenizer)
    assert len(tok_ids) == 2048
    sp = SamplingParams(max_tokens=NUM_GEN, temperature=0.0, ignore_eos=True)
    dense_alloc = torch.cuda.memory_allocated()
    dense_tok, dense_ttft, dense_dec = timed_run(dense, tok_ids, sp)
    dense_text = dense.tokenizer.decode(dense_tok)
    dense_paged = int(dense.model_runner.kv_cache.numel() *
                      dense.model_runner.kv_cache.element_size())
    dense.exit()

    sp_kwargs = dict(base_kwargs)
    sp_kwargs.update(enable_sparse_attention=True, sparse_selector="query_guided",
                     sparse_beta_raw=BETA_RAW, sparse_recent_tokens=128)
    sparse_eng = LLM(**sp_kwargs)
    alloc_after_sparse_init = torch.cuda.memory_allocated()
    sparse_paged = int(sparse_eng.model_runner.kv_cache.numel() *
                       sparse_eng.model_runner.kv_cache.element_size())

    results = {"config": {
        "model": MODEL, "prompt_tokens": 2048, "num_gen": NUM_GEN,
        "max_model_len": max_model_len, "beta_raw": BETA_RAW,
        "beta_scaled_logit": BETA_RAW / (HEAD_DIM ** 0.5),
        "recent_tokens": 128, "rbs": 64, "head_dim": HEAD_DIM,
    }, "methods": {}}
    results["dense_baseline"] = {
        "token_ids": dense_tok, "text": dense_text,
        "ttft_ms": dense_ttft, "decode_p50_ms": pct(dense_dec, 50),
        "decode_p95_ms": pct(dense_dec, 95),
        "total_ms": dense_ttft + sum(dense_dec)}

    for sel in SELECTORS:
        sparse_eng.sparse_reset()
        sparse_eng.sparse_set_selector(sel)
        torch.cuda.synchronize()
        alloc_before = torch.cuda.memory_allocated()
        cpu_hist = sparse_eng.sparse_history_bytes()
        tok, ttft, dec = timed_run(sparse_eng, tok_ids, sp)
        torch.cuda.synchronize()
        counters = sparse_eng.sparse_counters()
        sample = sparse_eng.sparse_sample_selection(14)
        text = sparse_eng.tokenizer.decode(tok)
        agree = sum(1 for a, b in zip(tok, dense_tok) if a == b)
        first_div = next((i for i, (a, b) in enumerate(zip(tok, dense_tok)) if a != b), len(tok))
        results["methods"][sel] = {
            "token_ids": tok, "text": text, "ttft_ms": ttft,
            "decode_p50_ms": pct(dec, 50), "decode_p95_ms": pct(dec, 95),
            "total_ms": ttft + sum(dec),
            "token_agreement": agree / max(1, len(dense_tok)),
            "first_divergence_position": first_div,
            "counters": counters, "selection_sample_layer14": sample,
            "cpu_history_bytes": cpu_hist,
            "cuda_allocated_bytes": alloc_before,
            "needle_present": "74291" in text,
        }

    results["memory"] = {
        "dense_paged_kv_bytes": dense_paged,
        "sparse_paged_kv_bytes": sparse_paged,
        "cuda_allocated_after_dense_init": dense_alloc,
        "cuda_allocated_after_sparse_init": alloc_after_sparse_init,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    }
    results["task"] = {"needle": "74291", "dense_found_needle": "74291" in dense_text}
    try:
        results["git_head_at_benchmark_start"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd="/opt/nano-vllm").decode().strip()
    except Exception:
        results["git_head_at_benchmark_start"] = "unknown"

    # abort assertions
    assert len(tok_ids) == 2048
    assert results["methods"]["full"]["token_agreement"] >= 0.99
    assert results["methods"]["query_guided"]["counters"]["dense_decode_fallbacks"] == 0
    assert results["memory"]["sparse_paged_kv_bytes"] == 0

    sparse_eng.exit()

    atomic_write(args.output_json,
                 json.dumps(results, indent=2, default=str))

    # CSV (explicit unix line endings)
    import csv
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["selector", "selected_token_ratio", "pre_window_recall",
                    "token_agreement", "ttft_ms", "decode_p50_ms",
                    "decode_p95_ms", "total_ms", "needle_present"])
        for n, m in results["methods"].items():
            s = m.get("selection_sample_layer14") or {}
            w.writerow([n, round(s.get("selected_token_ratio", 0), 3),
                        round((s.get("work") or {}).get("pre_window_recall", 0), 3),
                        round(m["token_agreement"], 3), round(m["ttft_ms"], 1),
                        round(m["decode_p50_ms"], 1), round(m["decode_p95_ms"], 1),
                        round(m["total_ms"], 1), m["needle_present"]])

    print("DENSE:", dense_text)
    for n, m in results["methods"].items():
        s = m.get("selection_sample_layer14") or {}
        print(n, "agree", round(m["token_agreement"], 2),
              "ratio", round(s.get("selected_token_ratio", 0), 2),
              "recall", round((s.get("work") or {}).get("pre_window_recall", 0), 2),
              "tp50", round(m["decode_p50_ms"], 1))


if __name__ == "__main__":
    main()
