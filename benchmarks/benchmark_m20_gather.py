"""CPU microbenchmark for token-wise versus whole-block decode KV gather.

This benchmark replays the real 32-block ID groups recorded by the M19 32K
decode run. It measures only the CPU gather methods; it does not run a model,
copy data to CUDA, or profile the process.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch
import torch.utils.benchmark as benchmark


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "bench_logs/m19_decode_raw_ab_3_0.json"
OUTPUT_PATH = ROOT / "bench_logs/m20_gather_micro_32k.json"

N_BLOCKS = 512
BLOCK_SIZE = 64
NUM_KV_HEADS = 8
HEAD_DIM = 128
TOP_K_BLOCKS = 32
NUM_LAYERS = 36
NUM_STEPS = 31
DTYPE = torch.bfloat16
THREAD_COUNTS = (1, 2, 4, 8)
ROUNDS = 3
MIN_RUN_TIME_S = 0.2


def load_real_ids() -> list[torch.Tensor]:
    with INPUT_PATH.open(encoding="utf-8") as f:
        record = json.load(f)
    history = record["payload"]["ids_history_by_layer"]
    assert len(history) == NUM_LAYERS, f"expected {NUM_LAYERS} layers, got {len(history)}"

    groups: list[torch.Tensor] = []
    for layer_id, layer_steps in enumerate(history):
        assert len(layer_steps) == NUM_STEPS, (
            f"layer {layer_id}: expected {NUM_STEPS} steps, got {len(layer_steps)}"
        )
        for step_id, ids in enumerate(layer_steps):
            assert len(ids) == TOP_K_BLOCKS, (
                f"layer {layer_id} step {step_id}: expected {TOP_K_BLOCKS} IDs, got {len(ids)}"
            )
            assert all(0 <= int(block_id) < N_BLOCKS for block_id in ids), (
                f"layer {layer_id} step {step_id}: block ID outside [0, {N_BLOCKS})"
            )
            groups.append(torch.tensor(ids, dtype=torch.long))
    assert len(groups) == NUM_LAYERS * NUM_STEPS
    return groups


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH,
                        help="use a distinct path to preserve earlier measurements")
    args = parser.parse_args()
    started = time.perf_counter()
    ids_groups = load_real_ids()

    # Match the runtime source geometry and keep it ordinary pageable CPU memory.
    source_shape = (N_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM)
    k_cpu = torch.empty(source_shape, dtype=DTYPE, device="cpu")
    v_cpu = torch.empty_like(k_cpu)
    assert not k_cpu.is_pinned() and not v_cpu.is_pinned()
    k_cpu.uniform_(-1, 1)
    v_cpu.uniform_(-1, 1)

    stage_shape = (TOP_K_BLOCKS * BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM)
    stage_k = torch.empty(stage_shape, dtype=DTYPE, device="cpu", pin_memory=True)
    stage_v = torch.empty_like(stage_k, pin_memory=True)
    ref_k = torch.empty(stage_shape, dtype=DTYPE, device="cpu")
    ref_v = torch.empty_like(ref_k)
    assert stage_k.is_pinned() and stage_v.is_pinned(), "staging buffers must be pinned"
    stage_ptrs = (stage_k.data_ptr(), stage_v.data_ptr())

    flat_k = k_cpu.view(-1, NUM_KV_HEADS, HEAD_DIM)
    flat_v = v_cpu.view(-1, NUM_KV_HEADS, HEAD_DIM)

    def gather_flat(ids: torch.Tensor) -> None:
        # Current path: construct token IDs for each selected block, then gather
        # the same token order into the preallocated pinned staging buffers.
        tok_idx = (
            ids.view(-1, 1) * BLOCK_SIZE
            + torch.arange(BLOCK_SIZE, dtype=torch.long)
        ).reshape(-1)
        torch.index_select(flat_k, 0, tok_idx, out=stage_k[:stage_shape[0]])
        torch.index_select(flat_v, 0, tok_idx, out=stage_v[:stage_shape[0]])

    def gather_blocks(ids: torch.Tensor) -> None:
        # Candidate path: preserve block order and gather along source dim 0.
        torch.index_select(
            k_cpu, 0, ids,
            out=stage_k.view(TOP_K_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM),
        )
        torch.index_select(
            v_cpu, 0, ids,
            out=stage_v.view(TOP_K_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM),
        )

    # Validate every recorded layer/step group before timing. Reuse fixed-size
    # reference buffers so the validation does not allocate one output per group.
    for group_index, ids in enumerate(ids_groups):
        gather_flat(ids)
        ref_k.copy_(stage_k)
        ref_v.copy_(stage_v)
        gather_blocks(ids)
        assert torch.equal(stage_k, ref_k), f"K mismatch for ID group {group_index}"
        assert torch.equal(stage_v, ref_v), f"V mismatch for ID group {group_index}"
        assert stage_k.is_pinned() and stage_v.is_pinned()
        assert stage_ptrs == (stage_k.data_ptr(), stage_v.data_ptr()), (
            f"staging storage changed for ID group {group_index}"
        )

    cursors = {"flat_token_index": 0, "block_dim0": 0}

    def flat_once() -> None:
        index = cursors["flat_token_index"]
        gather_flat(ids_groups[index])
        cursors["flat_token_index"] = (index + 1) % len(ids_groups)

    def block_once() -> None:
        index = cursors["block_dim0"]
        gather_blocks(ids_groups[index])
        cursors["block_dim0"] = (index + 1) % len(ids_groups)

    print(
        f"Validated {len(ids_groups)} real ID groups; exact K/V parity; "
        f"pinned stage pointers stable: {stage_ptrs}",
        flush=True,
    )

    functions = {"flat_token_index": flat_once, "block_dim0": block_once}
    raw_results: dict[str, dict[str, dict[str, object]]] = {}
    for thread_index, num_threads in enumerate(THREAD_COUNTS):
        raw_results[str(num_threads)] = {name: {"rounds": []} for name in functions}
        timers = {
            name: benchmark.Timer(
                stmt=f"{name}()",
                globals={name: fn},
                num_threads=num_threads,
                label=f"M20 gather {name}",
            )
            for name, fn in functions.items()
        }
        for round_index in range(ROUNDS):
            order = list(functions)
            if (thread_index + round_index) % 2:
                order.reverse()
            for name in order:
                measurement = timers[name].blocked_autorange(
                    min_run_time=MIN_RUN_TIME_S
                )
                round_record = {
                    "raw_times_seconds_per_batch": [float(value) for value in measurement.raw_times],
                    "normalized_times_seconds_per_gather": [
                        float(value / measurement.number_per_run)
                        for value in measurement.raw_times
                    ],
                    "number_per_run": int(measurement.number_per_run),
                    "timer_invocations_measured": len(measurement.raw_times)
                    * measurement.number_per_run,
                    "distinct_id_groups_covered_at_least": min(
                        len(ids_groups),
                        len(measurement.raw_times) * measurement.number_per_run,
                    ),
                    "median_seconds_per_gather": float(measurement.median),
                    "median_microseconds_per_gather": float(measurement.median * 1e6),
                }
                raw_results[str(num_threads)][name]["rounds"].append(round_record)
                print(
                    f"threads={num_threads} round={round_index + 1}/{ROUNDS} "
                    f"method={name} median={measurement.median * 1e3:.3f} ms/gather "
                    f"({measurement.median * 1e6:.2f} us/gather)",
                    flush=True,
                )

        for name in functions:
            rounds = raw_results[str(num_threads)][name]["rounds"]
            median_gather = statistics.median(
                float(round_record["median_seconds_per_gather"])
                for round_record in rounds
            )
            raw_results[str(num_threads)][name]["median_seconds_per_gather"] = median_gather
            raw_results[str(num_threads)][name]["median_microseconds_per_gather"] = (
                median_gather * 1e6
            )

        flat_median = float(raw_results[str(num_threads)]["flat_token_index"]["median_seconds_per_gather"])
        block_median = float(raw_results[str(num_threads)]["block_dim0"]["median_seconds_per_gather"])
        print(
            f"threads={num_threads} three-round-median: "
            f"flat={flat_median * 1e3:.3f} ms/gather, "
            f"block={block_median * 1e3:.3f} ms/gather, "
            f"speedup={flat_median / block_median:.3f}x",
            flush=True,
        )

    output = {
        "benchmark": "m20_cpu_decode_gather_microbenchmark",
        "input_ids": str(INPUT_PATH),
        "shape": {
            "source_k_v": list(source_shape),
            "stage_k_v": list(stage_shape),
            "layers": NUM_LAYERS,
            "decode_steps_per_layer": NUM_STEPS,
            "id_groups": len(ids_groups),
            "selected_blocks_per_group": TOP_K_BLOCKS,
            "dtype": str(DTYPE),
            "source_is_pinned": False,
            "stage_is_pinned": True,
        },
        "measurement": {
            "kind": "CPU gather microbenchmark; not TPOT or end-to-end latency",
            "timer": "torch.utils.benchmark.Timer.blocked_autorange",
            "min_run_time_seconds_per_round": MIN_RUN_TIME_S,
            "rounds_per_method_per_thread_count": ROUNDS,
            "method_order": "alternates across rounds and thread-count bands",
            "id_group_sampling": "one real layer-step group per invocation; each method cycles through all 1116 groups",
            "gather_invocation_bytes_k_plus_v": 2 * stage_k.numel() * stage_k.element_size(),
            "limits": [
                "One synthetic K/V source pair is reused for every real layer-step ID group; tensors do not vary by layer.",
                "blocked_autorange calibrates each method and round independently; method-specific cursors mean round input sequences are not strictly paired.",
                "CPU gather microbenchmark only; excludes model execution, CUDA transfer, attention, and TPOT effects.",
            ],
            "thread_counts": list(THREAD_COUNTS),
            "methods": ["flat_token_index", "block_dim0"],
            "elapsed_seconds_including_setup_validation_and_benchmark": time.perf_counter() - started,
        },
        "environment": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "cpu_threads_default": torch.get_num_threads(),
        },
        "results": raw_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote raw measurements and medians to {args.output}", flush=True)
    print(
        "WARNING: these are CPU gather microbenchmark results, not TPOT or end-to-end latency.",
        flush=True,
    )


if __name__ == "__main__":
    main()
