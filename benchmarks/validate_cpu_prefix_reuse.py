"""End-to-end Milestone 5 validation for CPU-backed prefix reuse."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoConfig

from nanovllm import LLM, SamplingParams
from nanovllm.kvdb.fingerprint import build_cache_fingerprint


def make_prompts() -> dict[str, list[int]]:
    base = [1000 + (index % 97) for index in range(600)]

    def changed_after(shared: int, offset: int, length: int = 600) -> list[int]:
        tokens = list(base[: min(shared, length)])
        tokens.extend(offset + (index % 89) for index in range(shared, length))
        return tokens

    first_diff = list(base)
    first_diff[0] = 9000
    return {
        "base": base,
        "gpu_partial": changed_after(256, 3000),
        "cpu_partial": changed_after(256, 5000),
        "non_aligned": changed_after(300, 7000),
        "first_block_diff": first_diff,
        "after_two_blocks": changed_after(512, 11000, length=900),
    }


def block_bytes(model: Path) -> int:
    config = AutoConfig.from_pretrained(model)
    fingerprint = build_cache_fingerprint(
        str(model), config, block_size=256, tensor_parallel_size=1
    )
    dtype_bytes = 2 if "16" in fingerprint.dtype else 4
    return (
        2
        * fingerprint.num_layers
        * fingerprint.block_size
        * fingerprint.num_kv_heads
        * fingerprint.head_dim
        * dtype_bytes
    )


def run_baseline(model: Path, output: Path) -> None:
    prompts = make_prompts()
    sampling = SamplingParams(temperature=0.0, max_tokens=2, ignore_eos=True)
    llm = LLM(
        str(model),
        enforce_eager=True,
        tensor_parallel_size=1,
        enable_cache_metrics=True,
        enable_reusable_cache=False,
        max_model_len=1024,
    )
    results = {}
    for name, prompt in prompts.items():
        generated = llm.generate([prompt], sampling, use_tqdm=False)[0]
        metrics = llm.get_cache_metrics()
        assert metrics["cpu_hit_blocks"] == 0
        assert metrics["reused_tokens"] == 0
        results[name] = {
            "prompt_tokens": len(prompt),
            "output_token_ids": generated["token_ids"],
            "metrics": metrics,
        }
        print(f"validated baseline: {name}", flush=True)
        llm.clear_gpu_prefix_cache()
    payload = {
        "phase": "feature_disabled_cold_baseline",
        "model": str(model),
        "cpu_cache_enabled": False,
        "sampling": {"temperature": 0.0, "max_tokens": 2, "ignore_eos": True},
        "prompts": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


def run_cpu(model: Path, baseline_path: Path, output: Path, pinned: bool) -> None:
    prompts = make_prompts()
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))["prompts"]
    sampling = SamplingParams(temperature=0.0, max_tokens=2, ignore_eos=True)
    capacity_blocks = 12
    llm = LLM(
        str(model),
        enforce_eager=True,
        tensor_parallel_size=1,
        enable_cache_metrics=True,
        enable_cpu_cache=True,
        cpu_cache_capacity_bytes=capacity_blocks * block_bytes(model),
        cpu_cache_pinned=pinned,
        max_model_len=1024,
    )
    cases = []

    def run_case(
        case: str,
        prompt_name: str,
        *,
        gpu_blocks: int,
        cpu_blocks: int,
        reused: int,
        recomputed: int,
        load_failures: int = 0,
    ):
        generated = llm.generate([prompts[prompt_name]], sampling, use_tqdm=False)[0]
        metrics = llm.get_cache_metrics()
        expected_ids = baseline[prompt_name]["output_token_ids"]
        assert generated["token_ids"] == expected_ids, (case, generated, expected_ids)
        assert metrics["gpu_hit_blocks"] == gpu_blocks, (case, metrics)
        assert metrics["cpu_hit_blocks"] == cpu_blocks, (case, metrics)
        assert metrics["reused_tokens"] == reused, (case, metrics)
        assert metrics["recomputed_tokens"] == recomputed, (case, metrics)
        assert metrics["prefill_executed_tokens"] == recomputed, (case, metrics)
        assert metrics["decode_executed_tokens"] == 1, (case, metrics)
        assert metrics["cpu_load_failures"] == load_failures, (case, metrics)
        cases.append(
            {
                "case": case,
                "prompt": prompt_name,
                "output_token_ids": generated["token_ids"],
                "metrics": metrics,
            }
        )
        print(f"validated CPU integration: {case}", flush=True)
        return metrics

    # A-C: cold, exact GPU hit, and GPU partial hit.
    run_case("A_cold_request", "base", gpu_blocks=0, cpu_blocks=0, reused=0, recomputed=600)
    run_case("B_exact_gpu_hit", "base", gpu_blocks=2, cpu_blocks=0, reused=512, recomputed=88)
    run_case("C_gpu_partial_hit", "gpu_partial", gpu_blocks=1, cpu_blocks=0, reused=256, recomputed=344)

    # D-H: clear GPU identity so every reuse must restore CPU bytes into newly
    # allocated physical blocks.
    gpu_cache_clear_counts = [llm.clear_gpu_prefix_cache()]
    assert gpu_cache_clear_counts[-1] > 0
    metrics = run_case("D_exact_cpu_hit", "base", gpu_blocks=0, cpu_blocks=2, reused=512, recomputed=88)
    assert metrics["h2d_bytes"] == 2 * block_bytes(model)

    gpu_cache_clear_counts.append(llm.clear_gpu_prefix_cache())
    run_case("E_cpu_partial_hit", "cpu_partial", gpu_blocks=0, cpu_blocks=1, reused=256, recomputed=344)

    gpu_cache_clear_counts.append(llm.clear_gpu_prefix_cache())
    run_case("F_non_aligned_prefix", "non_aligned", gpu_blocks=0, cpu_blocks=1, reused=256, recomputed=344)

    gpu_cache_clear_counts.append(llm.clear_gpu_prefix_cache())
    run_case("G_first_block_diff", "first_block_diff", gpu_blocks=0, cpu_blocks=0, reused=0, recomputed=600)

    gpu_cache_clear_counts.append(llm.clear_gpu_prefix_cache())
    run_case("H_diff_after_two_blocks", "after_two_blocks", gpu_blocks=0, cpu_blocks=2, reused=512, recomputed=388)

    # A forced load exception must release fresh GPU blocks and retry cold.
    gpu_cache_clear_counts.append(llm.clear_gpu_prefix_cache())
    original_restore = llm.model_runner.restore_cpu_blocks

    def fail_restore(_seqs):
        raise RuntimeError("injected CPU load failure")

    llm.model_runner.restore_cpu_blocks = fail_restore
    try:
        run_case(
            "fallback_after_injected_load_failure",
            "base",
            gpu_blocks=0,
            cpu_blocks=0,
            reused=0,
            recomputed=600,
            load_failures=1,
        )
        assert "injected CPU load failure" in llm.get_last_cpu_restore_error()
    finally:
        llm.model_runner.restore_cpu_blocks = original_restore

    # I: overflow the 12-block store with unique two-block contexts. The base
    # prefix was recently used, so seven contexts guarantee all prior blocks,
    # including base, have crossed the deterministic LRU boundary.
    for context_id in range(7):
        llm.clear_gpu_prefix_cache()
        prompt = [20000 + context_id * 500 + (i % 101) for i in range(600)]
        llm.generate([prompt], sampling, use_tqdm=False)
        print(f"filled LRU context: {context_id + 1}/7", flush=True)
    assert llm.get_cpu_cache_stats()["store"]["resident_blocks"] == capacity_blocks
    llm.clear_gpu_prefix_cache()
    run_case("I_cpu_lru_eviction", "base", gpu_blocks=0, cpu_blocks=0, reused=0, recomputed=600)

    block_manager = llm.scheduler.block_manager
    assert not block_manager.used_block_ids
    assert len(block_manager.free_block_ids) == len(block_manager.blocks)
    assert all(block.ref_count == 0 for block in block_manager.blocks)

    payload = {
        "phase": "cpu_backed_prefix_reuse",
        "model": str(model),
        "cpu_cache_enabled": True,
        "cpu_cache_pinned": pinned,
        "cpu_capacity_blocks": capacity_blocks,
        "cases": cases,
        "final_cpu_cache_stats": llm.get_cpu_cache_stats(),
        "gpu_cache_clear_counts": gpu_cache_clear_counts,
        "gpu_allocator_after_run": {
            "used_blocks": len(block_manager.used_block_ids),
            "free_blocks": len(block_manager.free_block_ids),
            "total_blocks": len(block_manager.blocks),
            "nonzero_ref_counts": sum(
                block.ref_count != 0 for block in block_manager.blocks
            ),
        },
        "case_J_feature_disabled": {
            "cpu_cache_enabled": False,
            "baseline_phase": baseline_path.name,
        },
        "feature_disabled_baseline": str(baseline_path),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--phase", choices=("baseline", "cpu"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--pinned", action="store_true")
    args = parser.parse_args()
    if args.phase == "baseline":
        run_baseline(args.model, args.output)
    else:
        if args.baseline is None:
            parser.error("--baseline is required for --phase cpu")
        run_cpu(args.model, args.baseline, args.output, args.pinned)


if __name__ == "__main__":
    main()
