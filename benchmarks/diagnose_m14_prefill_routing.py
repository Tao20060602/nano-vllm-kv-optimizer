"""Measure whether the final prefill router selects a known needle block."""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

from transformers import AutoTokenizer

from nanovllm import LLM, SamplingParams


HF_HOME = "/opt/models/.cache/huggingface"
YARN = {"rope_type": "yarn", "type": "yarn", "factor": 4.0,
        "original_max_position_embeddings": 32768, "rope_theta": 1000000}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query-segments", type=int, choices=(1, 4), required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    os.environ.setdefault("HF_HOME", HF_HOME)
    model = glob.glob(f"{HF_HOME}/hub/models--Qwen--Qwen3-4B/snapshots/*")[0]
    tok = AutoTokenizer.from_pretrained(model, use_fast=True)
    seqlen, target_at = 16384, 8192
    filler = tok.encode("The archive contains routine engineering notes. ")
    needle = tok.encode("The verification key is M14ORBIT42. ")
    question = tok.encode(" What is the verification key? Reply with only the key.")
    pre = (filler * ((target_at // len(filler)) + 1))[:target_at]
    prompt = pre + needle
    prompt += (filler * (((seqlen - len(prompt) - len(question)) // len(filler)) + 1))
    prompt = prompt[:seqlen - len(question)] + question
    assert len(prompt) == seqlen
    target_blocks = set(range(target_at // 64, (target_at + len(needle) - 1) // 64 + 1))

    llm = LLM(model, enforce_eager=True, tensor_parallel_size=1, max_num_seqs=1,
              max_model_len=32768, dtype="bfloat16", enable_sparse_attention=True,
              use_m12_runtime=True, sparse_selector="query_guided",
              sparse_retrieval_block_size=64, sparse_num_representatives=4,
              sparse_recent_tokens=512, sparse_first_tokens=64, sparse_top_k=32,
              sparse_prefill_chunk_size=4096,
              sparse_prefill_query_segments=args.query_segments,
              sparse_prefill_attention_backend="flash", rope_scaling_override=YARN)
    try:
        out = llm.generate([prompt], [SamplingParams(max_tokens=16, temperature=0.0,
                                                       ignore_eos=True)], use_tqdm=False)
        layers = [m.sparse_rt for m in llm.model_runner.model.modules()
                  if getattr(m, "sparse_rt", None) is not None]
        hits = [bool(target_blocks & set(layer.last_prefill_block_ids.tolist())) for layer in layers]
        result = {"query_segments": args.query_segments, "seq_len": seqlen,
                  "target_blocks": sorted(target_blocks), "layers": len(layers),
                  "target_block_selected_layers": sum(hits),
                  "target_block_selected_fraction": sum(hits) / len(hits),
                  "generated_text": tok.decode(out[0]["token_ids"])}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    finally:
        llm.exit()


if __name__ == "__main__":
    main()
