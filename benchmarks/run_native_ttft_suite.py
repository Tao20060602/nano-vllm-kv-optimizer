"""Run the frozen M23 baseline/direct-store audit and paired TTFT suite.

This orchestrator launches only the worker phases needed for the fixed-policy
comparison: one audit process per arm, followed by three fresh-process pairs in
baseline/direct_store, direct_store/baseline, baseline/direct_store order.
Every output is written beneath a new ignored ``bench_logs`` directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
PINNED_MODEL_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
EXPECTED_MODEL_FILE_COUNT = 13
EXPECTED_LENGTHS = (16480, 32864)
AUDIT_CHUNK_SIZE = 4096
AUDIT_LAYER_COUNT = 36
AUDIT_GENERATED_TOKEN_COUNT = 16
AUDIT_DECODE_TOP_K = 32
ARMS = ("baseline", "direct_store")
PERFORMANCE_ORDERS = (
    ("baseline", "direct_store"),
    ("direct_store", "baseline"),
    ("baseline", "direct_store"),
)

WORKER_EXTRA_SOURCES = (
    "benchmarks/benchmark_native_ttft.py",
    "benchmarks/native_ttft_candidate.py",
    "benchmarks/benchmark_segmented_adapter.py",
    "benchmarks/benchmark_m14_prefill.py",
    "docs/NATIVE_PREFILL_TTFT_OPTIMIZATION_PLAN.md",
)
SUITE_EXTRA_SOURCES = (
    *WORKER_EXTRA_SOURCES,
    "benchmarks/run_native_ttft_suite.py",
    "benchmarks/run_segmented_adapter_suite.py",
)

AUDIT_COMPONENTS = (
    "generated_ids",
    "attention_outputs",
    "scheduled",
    "states",
    "first_token_state",
    "final_logical_state",
    "decode_selections",
)
TIMING_FIELDS = (
    "ttft_ms",
    "prefill_step_sum_ms",
    "main_ms",
    "tail_ms",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_paths(extra_sources: tuple[str, ...]) -> list[Path]:
    package_files = sorted((ROOT / "nanovllm").rglob("*.py"))
    paths = package_files + [ROOT / name for name in extra_sources]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing source files: " + ", ".join(map(str, missing)))
    unique = {path.relative_to(ROOT).as_posix(): path for path in paths}
    return [unique[name] for name in sorted(unique)]


def source_hashes(extra_sources: tuple[str, ...]) -> dict[str, str]:
    return {
        path.relative_to(ROOT).as_posix(): sha256_file(path)
        for path in source_paths(extra_sources)
    }


def expected_worker_hashes(suite_hashes: dict[str, str]) -> dict[str, str]:
    worker_names = {
        path.relative_to(ROOT).as_posix()
        for path in source_paths(WORKER_EXTRA_SOURCES)
    }
    return {name: suite_hashes[name] for name in sorted(worker_names)}


def validate_python_environment() -> dict:
    expected_prefix = ROOT / ".venv"
    actual_prefix = Path(sys.prefix)
    if actual_prefix.resolve() != expected_prefix.resolve():
        raise RuntimeError(
            "run this suite with the checkout's .venv Python; "
            f"sys.prefix is {actual_prefix}, expected {expected_prefix}"
        )
    return {
        "executable": str(Path(sys.executable).absolute()),
        "prefix": str(actual_prefix.resolve()),
    }


def canonical_prompt_hash(prompt_ids: list[int]) -> str:
    encoded = json.dumps(prompt_ids, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_fixture(fixture_path: Path, model_manifest_sha256: str) -> tuple[dict, list[dict]]:
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    if fixture.get("status") != "frozen":
        raise ValueError("fixture status must be 'frozen'")
    if fixture.get("model_revision") != PINNED_MODEL_REVISION:
        raise ValueError("fixture does not identify the pinned Qwen3-4B revision")
    if fixture.get("model_manifest_sha256") != model_manifest_sha256:
        raise ValueError("fixture model manifest hash differs from the supplied model")

    requests = fixture.get("requests")
    if not isinstance(requests, list) or len(requests) != 4:
        raise ValueError("fixture must contain exactly four archive/code prompts")

    names: set[str] = set()
    combinations: set[tuple[str, int]] = set()
    validated = []
    for row in requests:
        if not isinstance(row, dict):
            raise ValueError("each fixture request must be a JSON object")
        name = row.get("name")
        prompt_ids = row.get("prompt_ids")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("fixture request names must be unique non-empty strings")
        names.add(name)
        if not isinstance(prompt_ids, list) or not prompt_ids:
            raise ValueError(f"{name}: prompt_ids must be a non-empty list")
        if any(type(token_id) is not int for token_id in prompt_ids):
            raise ValueError(f"{name}: prompt_ids must contain integer token IDs")
        length = len(prompt_ids)
        if length not in EXPECTED_LENGTHS:
            raise ValueError(f"{name}: expected {EXPECTED_LENGTHS} prompt tokens, got {length}")
        lowered = name.lower()
        if lowered.startswith("archive"):
            family = "archive"
        elif lowered.startswith("code"):
            family = "code"
        else:
            raise ValueError(f"{name}: request name must begin with archive or code")
        combination = (family, length)
        if combination in combinations:
            raise ValueError(f"fixture repeats input {family}/{length}")
        combinations.add(combination)
        actual_prompt_hash = canonical_prompt_hash(prompt_ids)
        if row.get("prompt_ids_sha256") != actual_prompt_hash:
            raise ValueError(f"{name}: prompt ID hash mismatch")
        validated.append({
            "name": name,
            "family": family,
            "tokens": length,
            "prompt_ids_sha256": actual_prompt_hash,
        })

    expected_combinations = {
        (family, length)
        for family in ("archive", "code")
        for length in EXPECTED_LENGTHS
    }
    if combinations != expected_combinations:
        raise ValueError("fixture must contain archive and code at both frozen lengths")
    return fixture, sorted(validated, key=lambda row: (row["family"], row["tokens"]))


def create_job(name: str, phase: str, arm: str, output_dir: Path, group: int | None = None) -> dict:
    return {
        "name": name,
        "phase": phase,
        "arm": arm,
        "group": group,
        "status": "pending",
        "worker_path": str((output_dir / f"{name}.json").resolve()),
        "log_path": str((output_dir / f"{name}.log").resolve()),
    }


def make_jobs(output_dir: Path) -> list[dict]:
    jobs = [
        create_job("audit-baseline", "audit", "baseline", output_dir),
        create_job("audit-direct_store", "audit", "direct_store", output_dir),
    ]
    for group in range(len(PERFORMANCE_ORDERS)):
        for arm in PERFORMANCE_ORDERS[group]:
            jobs.append(create_job(f"group{group}-{arm}", "perf", arm, output_dir, group))
    return jobs


def run_worker(
    job: dict,
    *,
    args,
    env: dict[str, str],
    record: dict,
    save,
    worker_hashes: dict[str, str],
) -> dict:
    worker_path = Path(job["worker_path"])
    log_path = Path(job["log_path"])
    if worker_path.exists() or log_path.exists():
        raise FileExistsError(f"refusing to overwrite job artifacts for {job['name']}")

    command = [
        sys.executable,
        str(ROOT / "benchmarks/benchmark_native_ttft.py"),
        "--model", str(args.model),
        "--fixture", str(args.fixture),
        "--arm", job["arm"],
        "--phase", job["phase"],
        "--output", str(worker_path),
    ]
    job.update(status="running", command=command, started_unix_s=time.time())
    save()
    started = time.perf_counter()
    try:
        with log_path.open("x", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
    except BaseException:
        job.update(status="failed", process_wall_s=time.perf_counter() - started)
        save()
        raise
    job.update(
        returncode=completed.returncode,
        process_wall_s=time.perf_counter() - started,
        finished_unix_s=time.time(),
    )
    if worker_path.is_file():
        job["worker_sha256"] = sha256_file(worker_path)
        try:
            worker = json.loads(worker_path.read_text(encoding="utf-8"))
        except Exception as exc:
            job.update(status="failed", worker_record_error=str(exc))
            save()
            raise RuntimeError(f"invalid worker record for {job['name']}") from exc
    else:
        worker = None

    if completed.returncode != 0:
        job.update(status="failed", worker_status=None if worker is None else worker.get("status"))
        save()
        raise RuntimeError(
            f"worker {job['name']} exited {completed.returncode}; log: {log_path}"
        )
    if worker is None:
        job.update(status="failed", worker_record_error="worker produced no JSON record")
        save()
        raise RuntimeError(f"worker {job['name']} produced no JSON record")

    try:
        validate_worker_record(worker, job, args, record, worker_hashes)
    except BaseException as exc:
        job.update(status="failed", validation_error=str(exc))
        save()
        raise
    job.update(
        status="passed",
        worker_status=worker["status"],
        candidate_provenance=worker["candidate"],
    )
    save()
    return worker


def validate_worker_record(
    worker: dict,
    job: dict,
    args,
    record: dict,
    worker_hashes: dict[str, str],
) -> None:
    if worker.get("status") != "passed":
        raise RuntimeError(f"worker {job['name']} did not pass: {worker.get('failure')}")
    if worker.get("arm") != job["arm"] or worker.get("phase") != job["phase"]:
        raise RuntimeError(f"worker {job['name']} arm/phase mismatch")
    if worker.get("source_hashes") != worker_hashes:
        raise RuntimeError(f"worker {job['name']} source hashes differ from suite baseline")
    if worker.get("source_hashes_after") != worker_hashes:
        raise RuntimeError(f"worker {job['name']} source files changed while it ran")
    if worker.get("fixture_sha256") != record["fixture_sha256"]:
        raise RuntimeError(f"worker {job['name']} fixture hash mismatch")
    if worker.get("model_manifest_sha256") != record["model_manifest_sha256"]:
        raise RuntimeError(f"worker {job['name']} model manifest hash mismatch")
    candidate = worker.get("candidate", {})
    if candidate.get("arm") != job["arm"]:
        raise RuntimeError(f"worker {job['name']} candidate installation mismatch")
    if candidate.get("runtime_count") != 36:
        raise RuntimeError(f"worker {job['name']} did not install across 36 runtimes")
    if candidate.get("layer_ids") != list(range(AUDIT_LAYER_COUNT)):
        raise RuntimeError(f"worker {job['name']} layer provenance is incomplete or out of order")
    if candidate.get("copy_non_blocking") is not False:
        raise RuntimeError(f"worker {job['name']} did not record blocking host copies")
    if job["arm"] == "baseline":
        if (
            candidate.get("replacement_count") != 0
            or candidate.get("replacements") != []
            or candidate.get("baseline_untouched") is not True
            or candidate.get("generated_function_sha256") != candidate.get("original_function_sha256")
        ):
            raise RuntimeError(f"worker {job['name']} baseline provenance mismatch")
    elif (
        candidate.get("replacement_count") != 2
        or len(candidate.get("replacements", [])) != 2
        or candidate.get("baseline_untouched") is not False
        or candidate.get("generated_function_sha256") == candidate.get("original_function_sha256")
    ):
        raise RuntimeError(f"worker {job['name']} direct-store provenance mismatch")


def normalize_attention_outputs(outputs: list[dict]) -> list[dict]:
    """Compare full output hashes while allowing contiguous metadata to differ."""
    ignored = {"k_contiguous", "v_contiguous"}
    return [{key: value for key, value in row.items() if key not in ignored} for row in outputs]


def requests_by_name(worker: dict, expected_names: set[str]) -> dict[str, dict]:
    rows = worker.get("requests")
    if not isinstance(rows, list):
        raise RuntimeError("worker request records are missing")
    mapping = {}
    for row in rows:
        name = row.get("name")
        if name in mapping:
            raise RuntimeError(f"worker returned duplicate request {name}")
        mapping[name] = row
    if set(mapping) != expected_names:
        raise RuntimeError(
            f"worker request names mismatch: expected {sorted(expected_names)}, got {sorted(mapping)}"
        )
    return mapping


def validate_audit_coverage(worker: dict, arm: str, case_names: list[str]) -> dict[str, dict]:
    """Require every measured attention output and decode selection to exist."""
    rows = requests_by_name(worker, set(case_names))
    coverage = {}
    expected_layers = set(range(AUDIT_LAYER_COUNT))
    for name in case_names:
        row = rows[name]
        tokens = row.get("tokens")
        generated_ids = row.get("generated_ids")
        if type(tokens) is not int or tokens <= 0:
            raise RuntimeError(f"{arm} audit has invalid token count for {name}")
        if (
            not isinstance(generated_ids, list)
            or len(generated_ids) != AUDIT_GENERATED_TOKEN_COUNT
            or any(type(token_id) is not int for token_id in generated_ids)
        ):
            raise RuntimeError(
                f"{arm} audit must contain exactly {AUDIT_GENERATED_TOKEN_COUNT} integer IDs for {name}"
            )

        prefill_steps = (tokens + AUDIT_CHUNK_SIZE - 1) // AUDIT_CHUNK_SIZE
        expected_step_count = prefill_steps + AUDIT_GENERATED_TOKEN_COUNT - 1
        expected_pairs = {
            (step, layer)
            for step in range(expected_step_count)
            for layer in expected_layers
        }
        outputs = row.get("attention_outputs")
        if not isinstance(outputs, list):
            raise RuntimeError(f"{arm} audit attention outputs are missing for {name}")
        observed_pairs = []
        for output in outputs:
            if not isinstance(output, dict):
                raise RuntimeError(f"{arm} audit has malformed attention output for {name}")
            step, layer = output.get("step"), output.get("layer")
            digest = output.get("sha256")
            if (
                type(step) is not int
                or type(layer) is not int
                or not isinstance(output.get("shape"), list)
                or not output["shape"]
                or any(type(dimension) is not int or dimension <= 0 for dimension in output["shape"])
                or not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise RuntimeError(f"{arm} audit has invalid attention output hash/key for {name}")
            observed_pairs.append((step, layer))
        observed_set = set(observed_pairs)
        if len(observed_pairs) != len(observed_set):
            raise RuntimeError(f"{arm} audit has duplicate attention step/layer output for {name}")
        missing = sorted(expected_pairs - observed_set)
        unexpected = sorted(observed_set - expected_pairs)
        if missing or unexpected:
            raise RuntimeError(
                f"{arm} audit attention coverage mismatch for {name}: "
                f"missing={missing[:8]}, unexpected={unexpected[:8]}"
            )

        decode_selections = row.get("decode_selections")
        if (
            not isinstance(decode_selections, list)
            or len(decode_selections) != AUDIT_LAYER_COUNT
            or any(
                not isinstance(layer_history, list)
                or len(layer_history) != AUDIT_GENERATED_TOKEN_COUNT - 1
                or any(
                    not isinstance(selection, list)
                    or len(selection) != AUDIT_DECODE_TOP_K
                    or any(type(block_id) is not int for block_id in selection)
                    for selection in layer_history
                )
                for layer_history in decode_selections
            )
        ):
            raise RuntimeError(
                f"{arm} audit must contain {AUDIT_GENERATED_TOKEN_COUNT - 1} decode selections "
                f"for each of {AUDIT_LAYER_COUNT} layers in {name}"
            )

        coverage[name] = {
            "tokens": tokens,
            "prefill_steps": prefill_steps,
            "attention_steps": expected_step_count,
            "attention_layers_per_step": AUDIT_LAYER_COUNT,
            "attention_pairs_expected": len(expected_pairs),
            "attention_pairs_observed": len(observed_pairs),
            "attention_coverage_complete": True,
            "decode_layers": len(decode_selections),
            "decode_selections_per_layer": AUDIT_GENERATED_TOKEN_COUNT - 1,
            "decode_ids_per_selection": AUDIT_DECODE_TOP_K,
        }
    return coverage


def compare_audit_workers(baseline: dict, candidate: dict, case_names: list[str]) -> dict:
    expected_names = set(case_names)
    baseline_coverage = validate_audit_coverage(baseline, "baseline", case_names)
    candidate_coverage = validate_audit_coverage(candidate, "direct_store", case_names)
    base_rows = requests_by_name(baseline, expected_names)
    candidate_rows = requests_by_name(candidate, expected_names)
    cases = []
    mismatches = []
    for name in case_names:
        base = base_rows[name]
        actual = candidate_rows[name]
        components = {}
        for component in AUDIT_COMPONENTS:
            left, right = base.get(component), actual.get(component)
            if component == "attention_outputs":
                if (
                    not isinstance(left, list)
                    or not isinstance(right, list)
                    or not left
                    or not right
                    or any(not isinstance(row, dict) or not isinstance(row.get("sha256"), str) for row in left)
                    or any(not isinstance(row, dict) or not isinstance(row.get("sha256"), str) for row in right)
                ):
                    raise RuntimeError(f"audit attention output hashes are missing for {name}")
                components[component] = normalize_attention_outputs(left or []) == normalize_attention_outputs(right or [])
            else:
                components[component] = left == right
        metadata_equal = [
            (row.get("k_contiguous"), row.get("v_contiguous"))
            for row in base.get("attention_outputs", [])
        ] == [
            (row.get("k_contiguous"), row.get("v_contiguous"))
            for row in actual.get("attention_outputs", [])
        ]
        row_result = {
            "name": name,
            "components": components,
            "attention_contiguity_metadata_equal": metadata_equal,
            "baseline_coverage": baseline_coverage[name],
            "direct_store_coverage": candidate_coverage[name],
            "passed": all(components.values()),
        }
        cases.append(row_result)
        for component, equal in components.items():
            if not equal:
                mismatches.append({"name": name, "component": component})
    return {
        "passed": not mismatches,
        "coverage": {"baseline": baseline_coverage, "direct_store": candidate_coverage},
        "cases": cases,
        "mismatches": mismatches,
    }


def assert_first_ids_match_audit(
    worker: dict,
    baseline_audit: dict,
    case_names: list[str],
) -> dict[str, int]:
    expected_names = set(case_names)
    references = requests_by_name(baseline_audit, expected_names)
    actual_rows = requests_by_name(worker, expected_names)
    matched = {}
    for name in case_names:
        expected = references[name].get("generated_ids")
        actual = actual_rows[name].get("generated_ids")
        if (
            not isinstance(expected, list)
            or len(expected) != 16
            or any(type(token_id) is not int for token_id in expected)
        ):
            raise RuntimeError(f"baseline audit lacks 16 token IDs for {name}")
        if (
            not isinstance(actual, list)
            or len(actual) != 1
            or type(actual[0]) is not int
        ):
            raise RuntimeError(f"performance worker must return exactly one ID for {name}")
        if actual[0] != expected[0]:
            raise RuntimeError(f"first generated token differs from baseline audit for {name}")
        matched[name] = int(actual[0])
    return matched


def finite_positive(value, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{label} must be finite and positive, got {value!r}")
    return number


def geometric_mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot compute a geometric mean of no values")
    checked = [finite_positive(value, "ratio") for value in values]
    return math.exp(math.fsum(math.log(value) for value in checked) / len(checked))


def ratio_stats(values: list[float]) -> dict:
    checked = [finite_positive(value, "ratio") for value in values]
    return {
        "geomean": geometric_mean(checked),
        "minimum": min(checked),
        "maximum": max(checked),
        "n": len(checked),
    }


def build_performance_summary(
    jobs: list[dict],
    worker_records: dict[str, dict],
    baseline_audit: dict,
    case_names: list[str],
) -> dict:
    rows_by_job = {
        name: requests_by_name(worker, set(case_names))
        for name, worker in worker_records.items()
    }
    pair_rows = []
    for group in range(len(PERFORMANCE_ORDERS)):
        baseline_name = f"group{group}-baseline"
        candidate_name = f"group{group}-direct_store"
        if baseline_name not in rows_by_job or candidate_name not in rows_by_job:
            raise RuntimeError(f"performance group {group} is incomplete")
        baseline_rows = rows_by_job[baseline_name]
        candidate_rows = rows_by_job[candidate_name]
        for case_name in case_names:
            base = baseline_rows[case_name]
            candidate = candidate_rows[case_name]
            base_ids = base.get("generated_ids")
            candidate_ids = candidate.get("generated_ids")
            if (
                not isinstance(base_ids, list)
                or len(base_ids) != 1
                or type(base_ids[0]) is not int
                or not isinstance(candidate_ids, list)
                or len(candidate_ids) != 1
                or type(candidate_ids[0]) is not int
            ):
                raise RuntimeError(f"group {group}/{case_name} did not generate one token")
            pair = {"group": group, "case": case_name}
            for field in TIMING_FIELDS:
                base_value = finite_positive(base[field], f"baseline {field}")
                candidate_value = finite_positive(candidate[field], f"candidate {field}")
                pair[f"{field}_baseline"] = base_value
                pair[f"{field}_direct_store"] = candidate_value
                pair[f"{field}_ratio"] = candidate_value / base_value
            base_steps = base.get("step_ms")
            candidate_steps = candidate.get("step_ms")
            expected_steps = (int(base["tokens"]) + 4095) // 4096
            if (
                not isinstance(base_steps, list)
                or not isinstance(candidate_steps, list)
                or len(base_steps) != expected_steps
                or len(candidate_steps) != expected_steps
            ):
                raise RuntimeError(f"group {group}/{case_name} step timing shape mismatch")
            pair["step_ms_baseline"] = [finite_positive(x, "baseline step") for x in base_steps]
            pair["step_ms_direct_store"] = [finite_positive(x, "candidate step") for x in candidate_steps]
            pair["step_ms_ratios"] = [
                candidate_value / base_value
                for base_value, candidate_value in zip(pair["step_ms_baseline"], pair["step_ms_direct_store"])
            ]
            pair["first_token_id"] = candidate_ids[0]
            pair_rows.append(pair)

    metric_summaries = {}
    for metric in TIMING_FIELDS:
        metric_summaries[metric] = {
            "all_12": ratio_stats([pair[f"{metric}_ratio"] for pair in pair_rows]),
            "by_group": {
                str(group): ratio_stats([
                    pair[f"{metric}_ratio"] for pair in pair_rows if pair["group"] == group
                ])
                for group in range(len(PERFORMANCE_ORDERS))
            },
            "by_case": {
                case: ratio_stats([pair[f"{metric}_ratio"] for pair in pair_rows if pair["case"] == case])
                for case in case_names
            },
        }
    max_steps = max(len(pair["step_ms_ratios"]) for pair in pair_rows)
    step_position_summaries = {
        str(position): ratio_stats([
            pair["step_ms_ratios"][position]
            for pair in pair_rows if len(pair["step_ms_ratios"]) > position
        ])
        for position in range(max_steps)
    }

    all_ttft = metric_summaries["ttft_ms"]["all_12"]["geomean"]
    every_group_gains = all(
        metric_summaries["ttft_ms"]["by_group"][str(group)]["geomean"] < 1.0
        for group in range(len(PERFORMANCE_ORDERS))
    )
    no_case_regresses = all(
        metric_summaries["ttft_ms"]["by_case"][case]["geomean"] <= 1.0
        for case in case_names
    )
    accepted = all_ttft <= 0.97 and every_group_gains and no_case_regresses
    return {
        "pairs": pair_rows,
        "candidate_over_baseline": metric_summaries,
        "step_position_ratios": step_position_summaries,
        "acceptance": {
            "threshold_all_12_geomean_le_0_97": all_ttft <= 0.97,
            "every_group_geomean_below_1": every_group_gains,
            "no_case_geomean_regression": no_case_regresses,
            "accepted": accepted,
        },
        "decision": "accepted" if accepted else "not_accepted",
    }


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="pinned local Qwen3-4B model")
    parser.add_argument("--fixture", type=Path, required=True, help="frozen four-input fixture JSON")
    parser.add_argument(
        "--output", type=Path, required=True,
        help="new directory below ignored bench_logs; existing directories are never overwritten",
    )
    return parser.parse_args(argv)


def run(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.model = args.model.expanduser().resolve()
    args.fixture = args.fixture.expanduser().resolve()
    output = args.output.expanduser()
    if not output.is_absolute():
        output = ROOT / output
    output = output.resolve()
    ignored_root = (ROOT / "bench_logs").resolve()
    if not output.is_relative_to(ignored_root) or output == ignored_root:
        raise ValueError("--output must be a new subdirectory below ignored bench_logs")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing suite directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()

    jobs = make_jobs(output)
    suite_path = output / "suite.json"
    record = {
        "schema": 1,
        "experiment": "M23 fixed-policy native TTFT direct-store candidate",
        "status": "running",
        "model_path": str(args.model),
        "fixture_path": str(args.fixture),
        "output_dir": str(output),
        "scope": "two audit processes, then three fresh-process performance pairs over four fixed prompts",
        "orders": [list(order) for order in PERFORMANCE_ORDERS],
        "jobs": jobs,
    }

    def save() -> None:
        suite_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    save()
    model_audit_fn = None
    model_audit_before = None
    model_manifest = args.model / ".cache/nanokv/source-manifest.json"
    try:
        record["python"] = validate_python_environment()
        record["source_hashes_before"] = source_hashes(SUITE_EXTRA_SOURCES)
        worker_hashes = expected_worker_hashes(record["source_hashes_before"])
        record["worker_source_hashes"] = worker_hashes
        if not args.model.is_dir():
            raise FileNotFoundError(f"model directory not found: {args.model}")
        if not args.fixture.is_file():
            raise FileNotFoundError(f"fixture file not found: {args.fixture}")
        if not model_manifest.is_file():
            raise FileNotFoundError(f"model source manifest not found: {model_manifest}")
        record["fixture_sha256"] = sha256_file(args.fixture)
        record["model_manifest_path"] = str(model_manifest)
        record["model_manifest_sha256"] = sha256_file(model_manifest)

        from run_segmented_adapter_suite import model_audit
        model_audit_fn = model_audit

        model_audit_before = model_audit(args.model)
        if model_audit_before.get("status") != "passed":
            raise RuntimeError("model audit did not pass")
        if len(model_audit_before.get("files", [])) != EXPECTED_MODEL_FILE_COUNT:
            raise RuntimeError(
                f"expected {EXPECTED_MODEL_FILE_COUNT} audited model files, "
                f"got {len(model_audit_before.get('files', []))}"
            )
        record["model_audit_before"] = model_audit_before

        fixture, fixture_inputs = validate_fixture(args.fixture, record["model_manifest_sha256"])
        case_names = [row["name"] for row in fixture_inputs]
        record["fixture_inputs"] = fixture_inputs
        if fixture.get("model_revision") != PINNED_MODEL_REVISION:
            raise RuntimeError("fixture revision changed during validation")

        child_env = os.environ.copy()
        for name in ("CUDA_HOME", "CUDA_PATH", "CUDACXX", "LD_LIBRARY_PATH"):
            child_env.pop(name, None)
        child_env.update(
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            OMP_NUM_THREADS="8",
            MKL_NUM_THREADS="8",
        )
        record["child_environment"] = {
            "python": record["python"]["executable"],
            "offline": True,
            "OMP_NUM_THREADS": "8",
            "MKL_NUM_THREADS": "8",
            "cleared_cuda_environment": ["CUDA_HOME", "CUDA_PATH", "CUDACXX", "LD_LIBRARY_PATH"],
        }

        workers: dict[str, dict] = {}
        baseline_audit = run_worker(
            jobs[0], args=args, env=child_env, record=record, save=save,
            worker_hashes=worker_hashes,
        )
        workers[jobs[0]["name"]] = baseline_audit
        candidate_audit = run_worker(
            jobs[1], args=args, env=child_env, record=record, save=save,
            worker_hashes=worker_hashes,
        )
        workers[jobs[1]["name"]] = candidate_audit
        audit_comparison = compare_audit_workers(baseline_audit, candidate_audit, case_names)
        record["audit_comparison"] = audit_comparison
        save()
        if not audit_comparison["passed"]:
            raise RuntimeError("baseline and direct_store audit records differ; performance phase skipped")

        baseline_first_ids = {
            name: int(row["generated_ids"][0])
            for name, row in requests_by_name(baseline_audit, set(case_names)).items()
        }
        record["audit_first_token_ids"] = baseline_first_ids
        save()

        for job in jobs[2:]:
            worker = run_worker(
                job, args=args, env=child_env, record=record, save=save,
                worker_hashes=worker_hashes,
            )
            workers[job["name"]] = worker
            matched_ids = assert_first_ids_match_audit(worker, baseline_audit, case_names)
            job["first_token_ids_match_audit"] = matched_ids
            save()

        record["performance_summary"] = build_performance_summary(
            jobs, workers, baseline_audit, case_names
        )

        if sha256_file(args.fixture) != record["fixture_sha256"]:
            raise RuntimeError("fixture changed during suite")
        if sha256_file(model_manifest) != record["model_manifest_sha256"]:
            raise RuntimeError("model manifest changed during suite")
        record["status"] = "passed"
        record["performance_decision"] = record["performance_summary"]["decision"]
    except BaseException as exc:
        record.update(status="failed", failure=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        if "source_hashes_before" in record:
            try:
                source_hashes_after = source_hashes(SUITE_EXTRA_SOURCES)
                record["source_hashes_after"] = source_hashes_after
                record["sources_unchanged"] = source_hashes_after == record["source_hashes_before"]
            except Exception as exc:
                record.setdefault("final_audit_errors", []).append(f"source hashes: {exc}")
        if args.fixture.is_file() and "fixture_sha256" in record:
            try:
                record["fixture_sha256_after"] = sha256_file(args.fixture)
                record["fixture_unchanged"] = record["fixture_sha256_after"] == record["fixture_sha256"]
            except Exception as exc:
                record.setdefault("final_audit_errors", []).append(f"fixture hash: {exc}")
        if model_manifest.is_file() and "model_manifest_sha256" in record:
            try:
                record["model_manifest_sha256_after"] = sha256_file(model_manifest)
                record["model_manifest_unchanged"] = (
                    record["model_manifest_sha256_after"] == record["model_manifest_sha256"]
                )
            except Exception as exc:
                record.setdefault("final_audit_errors", []).append(f"model manifest hash: {exc}")
        if model_audit_fn is not None and model_audit_before is not None:
            try:
                model_audit_after = model_audit_fn(args.model)
                record["model_audit_after"] = model_audit_after
                record["model_files_unchanged"] = model_audit_after == model_audit_before
            except Exception as exc:
                record.setdefault("final_audit_errors", []).append(f"model audit: {exc}")
        final_matches = all(record.get(key, True) for key in (
            "sources_unchanged", "fixture_unchanged", "model_manifest_unchanged", "model_files_unchanged"
        ))
        final_audit_failure = None
        if record.get("status") == "passed" and not final_matches:
            record.update(status="failed", failure="source/fixture/model changed during suite")
            final_audit_failure = RuntimeError(record["failure"])
        if record.get("final_audit_errors") and record.get("status") == "passed":
            record.update(status="failed", failure="final source/model audit failed")
            final_audit_failure = RuntimeError(record["failure"])
        save()
        print("native TTFT suite record", suite_path, flush=True)
        if final_audit_failure is not None:
            raise final_audit_failure


if __name__ == "__main__":
    run()
