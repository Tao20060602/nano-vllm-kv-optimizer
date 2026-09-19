from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from nanovllm import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser(description="Run a reproducible nano-vLLM latency and throughput baseline.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--num-requests", type=int, default=64)
    parser.add_argument("--min-input-length", type=int, default=64)
    parser.add_argument("--max-input-length", type=int, default=512)
    parser.add_argument("--min-output-length", type=int, default=32)
    parser.add_argument("--max-output-length", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark-results/baseline"))
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    if args.min_input_length > args.max_input_length:
        raise ValueError("min-input-length must not exceed max-input-length")
    if args.min_output_length > args.max_output_length:
        raise ValueError("min-output-length must not exceed max-output-length")

    rng = random.Random(args.seed)
    llm = LLM(
        args.model,
        enforce_eager=args.enforce_eager,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
    )

    vocab_upper_bound = min(10_000, llm.tokenizer.vocab_size - 1)
    prompts = [
        [rng.randint(0, vocab_upper_bound) for _ in range(rng.randint(args.min_input_length, args.max_input_length))]
        for _ in range(args.num_requests)
    ]
    sampling_params = [
        SamplingParams(
            temperature=args.temperature,
            ignore_eos=True,
            max_tokens=rng.randint(args.min_output_length, args.max_output_length),
        )
        for _ in range(args.num_requests)
    ]

    llm.generate([[1, 2, 3]], SamplingParams(temperature=args.temperature, max_tokens=4), use_tqdm=False)
    llm.generate(prompts, sampling_params, use_tqdm=False)
    metrics = llm.get_metrics()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(metrics["summary"], file, indent=2)
    write_csv(args.output_dir / "requests.csv", metrics["requests"])
    write_csv(args.output_dir / "steps.csv", metrics["steps"])

    print(json.dumps(metrics["summary"], indent=2))


if __name__ == "__main__":
    main()
