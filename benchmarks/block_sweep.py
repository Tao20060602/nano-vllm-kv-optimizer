"""Offline block-size replay sweep on real layer-14 K/V captured from one prefill.

For each retrieval block size (32/64/128/256) recompute mean-representative and
query-guided selection against the exact Block-DIPR oracle and report recall,
selected ratio, and work. Writes block_sweep.json.
"""
import json
import os
import time
from pathlib import Path

import torch
os.environ.setdefault("HF_HOME", "/opt/models/.cache/huggingface")
from nanovllm import LLM, SamplingParams
from nanovllm.sparse.block_sparse import exact_block_scores, select_dipr_blocks, gqa_token_scores
from nanovllm.sparse.representatives import BlockRepresentatives, flat_mean_select

OUT = Path("/opt/nano-vllm/benchmarks/results")


def main():
    eng = LLM("/opt/models/Qwen3-0.6B", enforce_eager=True, max_num_seqs=1,
              max_model_len=2096, enable_sparse_attention=True,
              sparse_selector="mean", sparse_beta_raw=48.0)
    sp = SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True)
    eng.sparse_reset()
    # prefill a real prompt; capture layer-14 CPU K/V
    eng.generate(["The secret passkey is 74291. " +
                  "The planet facts and star facts and orbit facts. " * 80],
                 [sp], use_tqdm=False)
    rt = None
    for m in eng.model_runner.model.modules():
        r = getattr(m, "sparse_rt", None)
        if r is not None and r.layer_id == 14:
            rt = r
    T = rt.valid_len
    k = rt.k_cpu[:T].float()  # [T,Hkv,D]
    q_last = rt.q_samples[-1]  # [Hq,D] real non-last prefill query
    eng.exit()

    rows = []
    for rbs in (32, 64, 128, 256):
        reps = BlockRepresentatives(k, rbs, 4)
        q = q_last.float()
        exact = exact_block_scores(gqa_token_scores(q, k), rbs)
        oracle = select_dipr_blocks(exact, 48.0).any(0)
        n_blocks = oracle.shape[0]
        res = flat_mean_select(q, reps, 48.0)
        sel = res.per_head_mask.any(0)[:n_blocks]
        L = min(sel.shape[0], oracle.shape[0])
        rec = float((sel[:L] & oracle[:L]).sum()) / max(1, int(oracle[:L].sum()))
        rows.append({"rbs": rbs, "blocks": int(n_blocks),
                     "oracle_blocks": int(oracle[:L].sum()),
                     "recall": round(rec, 3),
                     "selected_ratio": round(float(sel[:L].sum()) / n_blocks, 3),
                     "refine_pairs": res.refined_block_count})
    out = {"block_size_sweep": rows}
    (OUT / "block_sweep.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
