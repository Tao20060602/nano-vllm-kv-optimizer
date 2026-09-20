import argparse
import json
import os
from pathlib import Path
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=os.environ.get(
            "NANOVLLM_MODEL", os.path.expanduser("~/huggingface/Qwen3-0.6B/")
        ),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    path = args.model
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    if args.output is not None:
        result = {
            "model": path,
            "temperature": sampling_params.temperature,
            "max_tokens": sampling_params.max_tokens,
            "outputs": [
                {
                    "prompt": prompt,
                    "text": output["text"],
                    "token_ids": output["token_ids"],
                }
                for prompt, output in zip(prompts, outputs)
            ],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
