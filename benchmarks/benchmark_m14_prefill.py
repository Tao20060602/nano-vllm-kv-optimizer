"""M14 later-chunk prefill benchmark with reproducible, synchronized timing.

This intentionally reports a prefill wall time, not decode TPOT.  It runs one
real Qwen3-4B request, synchronizes around ``generate``, and persists the
configuration alongside the result so torch-reference and FlashAttention-2
experiments can be compared without mixing their semantics.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import time
from pathlib import Path

import psutil
import torch
from transformers import AutoTokenizer

from nanovllm import LLM, SamplingParams


HF_HOME = "/opt/models/.cache/huggingface"
YARN = {
    "rope_type": "yarn",
    "type": "yarn",
    "factor": 4.0,
    "original_max_position_embeddings": 32768,
    "rope_theta": 1000000,
}


def make_prompt(tokenizer: AutoTokenizer, seqlen: int) -> list[int]:
    unit = tokenizer.encode(
        "The archive records that every experiment needs a reproducible "
        "configuration and a clear measurement boundary. "
    )
    ids = (unit * ((seqlen // len(unit)) + 1))[:seqlen]
    assert len(ids) == seqlen
    return ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--query-segments", type=int, default=4)
    parser.add_argument("--backend", choices=("torch", "flash"), default="flash")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert args.seq_len > args.chunk_size > 0

    os.environ.setdefault("HF_HOME", HF_HOME)
    model = glob.glob(
        f"{HF_HOME}/hub/models--Qwen--Qwen3-4B/snapshots/*"
    )[0]
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    prompt_ids = make_prompt(tokenizer, args.seq_len)
    llm = LLM(
        model,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_num_seqs=1,
        max_model_len=max(16384, args.seq_len + 16),
        dtype="bfloat16",
        enable_sparse_attention=True,
        use_m12_runtime=True,
        sparse_selector="query_guided",
        sparse_retrieval_block_size=64,
        sparse_num_representatives=4,
        sparse_recent_tokens=512,
        sparse_first_tokens=64,
        sparse_top_k=32,
        sparse_prefill_chunk_size=args.chunk_size,
        sparse_prefill_query_segments=args.query_segments,
        sparse_prefill_attention_backend=args.backend,
        rope_scaling_override=YARN,
    )
    try:
        process = psutil.Process()
        rss_before = process.memory_info().rss
        torch.cuda.reset_peak_memory_stats()
        llm.add_request(
            prompt_ids,
            SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True),
        )
        prefill_step_ms = []
        decode_step_ms = []
        output = []
        while not llm.is_finished():
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            output, num_scheduled_tokens = llm.step()
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            if num_scheduled_tokens > 0:
                prefill_step_ms.append(elapsed_ms)
            else:
                decode_step_ms.append(elapsed_ms)

        layers = [
            module.sparse_rt
            for module in llm.model_runner.model.modules()
            if getattr(module, "sparse_rt", None) is not None
        ]
        assert len(layers) == 36
        valid_lens = {layer.valid_len for layer in layers}
        # Depending on the scheduler's first-token bookkeeping, generate(1)
        # may leave sparse history at prompt length or prompt+one.  Either is
        # valid here; require a complete, internally consistent prompt state.
        assert len(valid_lens) == 1 and next(iter(valid_lens)) >= args.seq_len
        assert all(torch.isfinite(layer.reps_gpu[:, :layer.nblocks_filled]).all()
                   for layer in layers)
        result = {
            "measurement": (
                "CUDA-synchronized engine steps; max_tokens=1 samples the first "
                "token in the final prefill step and has no standalone decode step"
            ),
            "prefill_wall_ms": sum(prefill_step_ms),
            "prefill_step_ms": prefill_step_ms,
            "decode_step_ms": decode_step_ms,
            "config": {
                "model": model,
                "seq_len": args.seq_len,
                "chunk_size": args.chunk_size,
                "query_segments": args.query_segments,
                "prefill_attention_backend": args.backend,
                "block_size": 64,
                "representatives": 4,
                "top_k_blocks": 32,
                "sink_tokens": 64,
                "recent_tokens": 512,
            },
            "generated_tokens": len(output[0][1]) if output else 0,
            "runtime": {
                "layers": len(layers),
                "layer0_valid_len": layers[0].valid_len,
                "layer0_last_prefill_query_summaries": layers[0].last_prefill_query_summaries,
                "host_rss_before_gib": rss_before / 1024**3,
                "host_rss_after_gib": process.memory_info().rss / 1024**3,
                "host_available_after_gib": psutil.virtual_memory().available / 1024**3,
                "gpu_allocated_gib": torch.cuda.memory_allocated() / 1024**3,
                "gpu_reserved_gib": torch.cuda.memory_reserved() / 1024**3,
                "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
            },
            "git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.output.with_suffix(args.output.suffix + ".tmp")
        tmp.write_text(json.dumps(result, indent=2) + "\n")
        tmp.replace(args.output)
        print(json.dumps(result, indent=2))
    finally:
        llm.exit()


if __name__ == "__main__":
    main()
