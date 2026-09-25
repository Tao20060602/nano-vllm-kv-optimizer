"""Five-case needle diagnostic for the repaired M14 path.

The answer-substring hit is a debugging signal, not a model-quality benchmark.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from nanovllm import LLM, SamplingParams


HF_HOME = "/opt/models/.cache/huggingface"
YARN = {"rope_type": "yarn", "type": "yarn", "factor": 4.0,
        "original_max_position_embeddings": 32768, "rope_theta": 1000000}


def contains(whole: list[int], part: list[int]) -> bool:
    return any(whole[i:i + len(part)] == part
               for i in range(len(whole) - len(part) + 1))


def place(tokens: list[int], payload: list[int], position: int) -> None:
    assert 0 <= position <= len(tokens) - len(payload)
    tokens[position:position + len(payload)] = payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=65536)
    parser.add_argument("--query-segments", type=int, choices=(1, 4, 64), required=True)
    parser.add_argument("--case", default="all")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault("HF_HOME", HF_HOME)
    model = glob.glob(f"{HF_HOME}/hub/models--Qwen--Qwen3-4B/snapshots/*")[0]
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    filler = tokenizer.encode(
        "The quick brown fox jumps over the lazy dog near the calm river bank. ")
    cases = [
        ("simple-10", "The secret code is NANOAlphaKilo. ",
         "What is the secret code?", "NANOAlphaKilo", 0.10, ()),
        ("simple-50", "The secret code is NANOCharlieKilo. ",
         "What is the secret code?", "NANOCharlieKilo", 0.50, ()),
        ("simple-90", "The secret code is NANOEchoKilo. ",
         "What is the secret code?", "NANOEchoKilo", 0.90, ()),
        ("distract-50", "The secret code is NANOFoxKilo. ",
         "What is the secret code?", "NANOFoxKilo", 0.50,
         ("The secret code is NANOWRONGKilo. ",
          "The magic number is 12345. ", "The magic number is 67890. ")),
        ("multikey-50",
         "The secret code is NANOGolfKilo and the backup code is ZULU42. ",
         "What is the backup code?", "ZULU42", 0.50, ()),
    ]
    if args.case != "all":
        cases = [case for case in cases if case[0] == args.case]
        if not cases:
            raise ValueError(f"unknown case: {args.case}")
    result = {"config": {"model": model, "seq_len": args.seq_len,
                           "query_segments": args.query_segments,
                           "chunk_size": 4096, "top_k_blocks": 32,
                           "representatives": 4}, "cases": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for name, needle, question, answer, fraction, distractors in cases:
        prompt = (filler * ((args.seq_len // len(filler)) + 1))[:args.seq_len]
        needle_ids = tokenizer.encode(needle)
        question_ids = tokenizer.encode(question)
        target = min(int(args.seq_len * fraction),
                     args.seq_len - len(question_ids) - len(needle_ids) - 32)
        place(prompt, needle_ids, target)
        for offset, distractor in zip((0.18, 0.32, 0.72), distractors):
            place(prompt, tokenizer.encode(distractor), int(args.seq_len * offset))
        place(prompt, question_ids, args.seq_len - len(question_ids))
        assert contains(prompt, needle_ids) and contains(prompt, question_ids)

        llm = LLM(
            model, enforce_eager=True, tensor_parallel_size=1, max_num_seqs=1,
            max_model_len=131072, dtype="bfloat16", enable_sparse_attention=True,
            use_m12_runtime=True, sparse_selector="query_guided",
            sparse_retrieval_block_size=64, sparse_num_representatives=4,
            sparse_recent_tokens=512, sparse_first_tokens=64, sparse_top_k=32,
            sparse_prefill_chunk_size=4096,
            sparse_prefill_query_segments=args.query_segments,
            sparse_prefill_attention_backend="flash",
            sparse_gather_index_select=True, rope_scaling_override=YARN,
        )
        try:
            torch.cuda.synchronize()
            start = time.perf_counter()
            output = llm.generate(
                [prompt], [SamplingParams(max_tokens=args.max_tokens,
                                          temperature=0.0, ignore_eos=True)],
                use_tqdm=False)
            torch.cuda.synchronize()
            wall_s = time.perf_counter() - start
            token_ids = list(output[0]["token_ids"])
            text = tokenizer.decode(token_ids)
            needle_block_id = target // 64
            selector_layers = [
                module.sparse_rt
                for module in llm.model_runner.model.modules()
                if getattr(module, "sparse_rt", None) is not None
            ]
            selected_needle_layers = sum(
                needle_block_id in layer.last_prefill_block_ids.tolist()
                for layer in selector_layers
                if layer.last_prefill_block_ids is not None
            )
            case_result = {"name": name, "needle_token_position": target,
                           "needle_present": contains(prompt, needle_ids),
                           "question_present": contains(prompt, question_ids),
                           "needle_block_id": needle_block_id,
                           "needle_block_selected_layers": selected_needle_layers,
                           "selector_layers": len(selector_layers),
                           "expected": answer, "hit": answer in text,
                           "generated_token_ids": token_ids,
                           "generated_text": text, "wall_s": wall_s}
            result["cases"].append(case_result)
            result["hits"] = sum(case["hit"] for case in result["cases"])
            temporary = args.output.with_suffix(args.output.suffix + ".tmp")
            temporary.write_text(json.dumps(result, indent=2) + "\n")
            temporary.replace(args.output)
            print(json.dumps(case_result, ensure_ascii=False), flush=True)
        finally:
            llm.exit()
            del llm
            torch.cuda.empty_cache()

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
