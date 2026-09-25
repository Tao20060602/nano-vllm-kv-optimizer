"""Run NanoKV on an already generated NVIDIA RULER JSONL file.

Supports two NVIDIA formats: the legacy RULER ``main`` scripts format
(``input``, ``outputs``, optional ``answer_prefix``; output prediction key
``pred``) and RULERv1's NeMo-Skills preparation format from
``NVIDIA-NeMo/Skills`` ``main`` (``question``, ``expected_answer``,
``generation``; output prediction key ``generation``). In the latter,
``generation`` is the assistant prefix placed after the Qwen chat-formatted
question. The default disables Qwen3 thinking to keep short task token budgets
focused on scored answers; ``--thinking-mode`` records and controls this choice.
The current
RULERv1 README directs users to NeMo-Skills and no longer uses the legacy
``scripts/pred/call_api.py`` path. The NeMo-Skills ``ns eval`` pipeline is not
installed or executed here, so matching its row format does not claim its
end-to-end evaluation was verified. Both upstream branches are mutable: pin
the exact RULER and NeMo-Skills commits in any formal experiment record. This
adapter deliberately does not generate or download data.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Any

from nanovllm import LLM, SamplingParams


HF_HOME = "/opt/models/.cache/huggingface"
YARN = {
    "rope_type": "yarn",
    "type": "yarn",
    "factor": 4.0,
    "original_max_position_embeddings": 32768,
    "rope_theta": 1000000,
}


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"expected a JSON object on {path}:{line_number}")
            is_legacy = isinstance(row.get("input"), str)
            is_rulerv1 = (isinstance(row.get("question"), str)
                          and isinstance(row.get("generation"), str))
            if not (is_legacy or is_rulerv1):
                raise ValueError(
                    f"expected legacy 'input' or RULERv1 'question' plus 'generation' "
                    f"on {path}:{line_number}")
            if is_legacy and is_rulerv1:
                raise ValueError(
                    f"ambiguous mixed legacy/RULERv1 fields on {path}:{line_number}")
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise ValueError(f"no examples found in {path}")
    return rows


def resolve_model(model_arg: str | None) -> str:
    if model_arg:
        return model_arg
    matches = sorted(glob.glob(
        f"{HF_HOME}/hub/models--Qwen--Qwen3-4B/snapshots/*"))
    if not matches:
        raise FileNotFoundError(
            f"Qwen3-4B not found under {HF_HOME}/hub; pass --model with a local model path")
    return matches[-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate RULER-compatible predictions with NanoKV (one prompt at a time).")
    parser.add_argument("--input", type=Path, required=True, help="generated RULER JSONL")
    parser.add_argument("--output", type=Path, required=True, help="prediction JSONL for RULER evaluator")
    parser.add_argument("--limit", type=int,
                        help="run only the first N rows (same input prefix for matched comparisons)")
    parser.add_argument("--dense", action="store_true",
                        help="run the dense baseline without passing sparse runtime options")
    parser.add_argument("--model", help="local model path; defaults to cached Qwen3-4B")
    parser.add_argument("--top-k", type=int, default=32,
                        help="decode history block cap (1-32); prefill remains at 32")
    parser.add_argument("--dynamic-top-k", action="store_true",
                        help="adapt decode K using the representative-score heuristic")
    parser.add_argument("--dynamic-mass", type=float, default=0.90,
                        help="target normalized selector-score mass for dynamic K")
    parser.add_argument("--max-tokens", type=int, required=True,
                        help="greedy token limit; set per RULER task (v1 defaults vary by task)")
    parser.add_argument(
        "--thinking-mode", choices=("disabled", "default", "enabled"),
        default="disabled",
        help=("Qwen chat-template thinking behavior for RULERv1; disabled is the "
              "default so short task budgets go to the scored answer"),
    )
    args = parser.parse_args()
    if not 1 <= args.top_k <= 32:
        parser.error("--top-k must be between 1 and 32 because prefill is fixed at 32")
    if not 0.0 < args.dynamic_mass <= 1.0:
        parser.error("--dynamic-mass must be in (0, 1]")
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.input.resolve() == args.output.resolve():
        parser.error("--output must not overwrite --input")
    return args


def main() -> None:
    args = parse_args()
    os.environ.setdefault("HF_HOME", HF_HOME)
    model = resolve_model(args.model)
    rows = read_jsonl(args.input, limit=args.limit)

    llm_config = dict(
        enforce_eager=True,
        tensor_parallel_size=1,
        max_num_seqs=1,
        max_model_len=131072,
        dtype="bfloat16",
        rope_scaling_override=YARN,
    )
    if not args.dense:
        llm_config.update(
            enable_sparse_attention=True,
            use_m12_runtime=True,
            sparse_selector="query_guided",
            sparse_retrieval_block_size=64,
            sparse_num_representatives=4,
            sparse_recent_tokens=512,
            sparse_first_tokens=64,
            sparse_top_k=32,
            sparse_decode_top_k=args.top_k,
            sparse_dynamic_top_k=args.dynamic_top_k,
            sparse_dynamic_top_k_mass=args.dynamic_mass,
            sparse_prefill_chunk_size=4096,
            sparse_prefill_query_segments=1,
            sparse_prefill_attention_backend="flash",
            sparse_gather_index_select=True,
        )
    llm = LLM(model, **llm_config)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    sampling = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=0.0,
        ignore_eos=False,
    )
    try:
        with temporary.open("w", encoding="utf-8", buffering=1) as destination:
            for row_number, source_row in enumerate(rows, start=1):
                if "input" in source_row:
                    prompt = source_row["input"] + source_row.get("answer_prefix", "")
                    prediction_key = "pred"
                else:
                    template_kwargs = {}
                    if args.thinking_mode != "default":
                        template_kwargs["enable_thinking"] = (
                            args.thinking_mode == "enabled")
                    formatted_question = llm.tokenizer.apply_chat_template(
                        [{"role": "user", "content": source_row["question"]}],
                        tokenize=False,
                        add_generation_prompt=True,
                        **template_kwargs,
                    )
                    prompt = formatted_question + source_row["generation"]
                    prediction_key = "generation"
                generated = llm.generate([prompt], sampling, use_tqdm=False)
                prediction = dict(source_row)
                prediction[prediction_key] = generated[0]["text"]
                destination.write(json.dumps(prediction, ensure_ascii=False) + "\n")
                print(f"[{row_number}/{len(rows)}] index={source_row.get('index', row_number - 1)}")
        temporary.replace(args.output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        llm.exit()

    print(json.dumps({
        "input": str(args.input),
        "output": str(args.output),
        "model": model,
        "rows": len(rows),
        "limit": args.limit,
        "prefill_top_k": None if args.dense else 32,
        "decode_top_k_cap": None if args.dense else args.top_k,
        "dynamic_top_k": False if args.dense else args.dynamic_top_k,
        "dynamic_mass": None if args.dense else args.dynamic_mass,
        "max_tokens": args.max_tokens,
        "sampling": "greedy",
        "ignore_eos": False,
        "thinking_mode": args.thinking_mode,
        "input_format": "rulerv1-ns" if "question" in rows[0] else "legacy-ruler",
        "dense": args.dense,
    }, indent=2))


if __name__ == "__main__":
    main()
