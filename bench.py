import argparse
import json
import os
import time
from pathlib import Path
from random import randint, seed
from nanovllm import LLM, SamplingParams
# from vllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=os.environ.get(
            "NANOVLLM_MODEL", os.path.expanduser("~/huggingface/Qwen3-0.6B/")
        ),
    )
    parser.add_argument("--num-seqs", type=int, default=256)
    parser.add_argument("--max-input-len", type=int, default=1024)
    parser.add_argument("--max-output-len", type=int, default=1024)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    seed(0)
    num_seqs = args.num_seqs
    max_input_len = args.max_input_len
    max_output_len = args.max_output_len

    path = args.model
    llm = LLM(path, enforce_eager=False, max_model_len=4096)

    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_output_len)) for _ in range(num_seqs)]
    # uncomment the following line for vllm
    # prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]

    llm.generate(["Benchmark: "], SamplingParams())
    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    t = (time.time() - t)
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t
    print(f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")
    if args.output is not None:
        result = {
            "model": path,
            "seed": 0,
            "num_seqs": num_seqs,
            "max_input_len": max_input_len,
            "max_output_len": max_output_len,
            "total_output_tokens": total_tokens,
            "elapsed_seconds": t,
            "throughput_tokens_per_second": throughput,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
