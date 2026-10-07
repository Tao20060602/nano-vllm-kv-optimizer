"""Independently audit and recompute a completed native TTFT evidence bundle.

This checker uses only the Python standard library. It does not import the
benchmark driver, suite helpers, PyTorch, or CUDA modules. It reads a completed
suite record and its raw worker JSONs, then verifies their hashes, provenance,
audit coverage, exact outputs, and pre-registered performance gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import traceback


ROOT = Path(__file__).resolve().parents[1]
PINNED_MODEL_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
EXPECTED_MODEL_FILE_COUNT = 13
EXPECTED_CASE_LENGTHS = (16480, 32864)
LAYER_COUNT = 36
PREFILL_CHUNK = 4096
AUDIT_GENERATED_TOKENS = 16
AUDIT_DECODE_STEPS = AUDIT_GENERATED_TOKENS - 1
TOP_K = 32
TIMING_FIELDS = ("ttft_ms", "prefill_step_sum_ms", "main_ms", "tail_ms")

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
EXPECTED_ORDERS = (
    ("baseline", "direct_store"),
    ("direct_store", "baseline"),
    ("baseline", "direct_store"),
)
EXPECTED_JOBS = (
    ("audit-baseline", "audit", "baseline", None),
    ("audit-direct_store", "audit", "direct_store", None),
    ("group0-baseline", "perf", "baseline", 0),
    ("group0-direct_store", "perf", "direct_store", 0),
    ("group1-direct_store", "perf", "direct_store", 1),
    ("group1-baseline", "perf", "baseline", 1),
    ("group2-baseline", "perf", "baseline", 2),
    ("group2-direct_store", "perf", "direct_store", 2),
)
EXPECTED_REPLACEMENTS = [
    {
        "before": "self.k_cpu[start_block:need].copy_(k_blocks.cpu())",
        "after": "self.k_cpu[start_block:need].copy_(k_blocks, non_blocking=False)",
    },
    {
        "before": "self.v_cpu[start_block:need].copy_(v_blocks.cpu())",
        "after": "self.v_cpu[start_block:need].copy_(v_blocks, non_blocking=False)",
    },
]
EXPECTED_CONFIG = {
    "block": 64,
    "representatives": 4,
    "top_k": TOP_K,
    "query_summaries": 1,
    "sink": 64,
    "recent": 512,
    "chunk": PREFILL_CHUNK,
    "backend": "flash",
    "index_select": True,
    "threads": 8,
    "selector_graph": False,
    "static_mask": False,
    "decode_pipeline": False,
}
HASH_LENGTH = 64
HEX = frozenset("0123456789abcdef")


class EvidenceError(RuntimeError):
    """A completed evidence bundle failed an independent consistency check."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_blob_sha1(path: Path) -> str:
    size = path.stat().st_size
    digest = hashlib.sha1()
    digest.update(b"blob " + str(size).encode("ascii") + b"\0")
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise EvidenceError(f"cannot read JSON {path}: {exc}") from exc


def is_sha256(value) -> bool:
    return (
        isinstance(value, str)
        and len(value) == HASH_LENGTH
        and all(character in HEX for character in value)
    )


def require_sha256(value, label: str) -> str:
    require(is_sha256(value), f"{label} is not a lowercase SHA-256 digest")
    return value


def resolve_path(value, label: str) -> Path:
    require(isinstance(value, str) and value, f"{label} path is missing")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def path_inside(path: Path, root: Path, label: str) -> Path:
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(root.resolve(strict=True))
    except ValueError as exc:
        raise EvidenceError(f"{label} escapes its evidence directory: {resolved}") from exc
    return resolved


def canonical_prompt_sha256(prompt_ids: list[int]) -> str:
    encoded = json.dumps(prompt_ids, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def source_names(extra_sources: tuple[str, ...]) -> list[str]:
    package = sorted(
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "nanovllm").rglob("*.py")
        if path.is_file()
    )
    names = sorted(set(package).union(extra_sources))
    missing = [name for name in names if not (ROOT / name).is_file()]
    require(not missing, "required source files are missing: " + ", ".join(missing))
    return names


def validate_source_evidence(suite: dict, workers: dict[str, dict]) -> dict:
    worker_names = source_names(WORKER_EXTRA_SOURCES)
    suite_names = source_names(SUITE_EXTRA_SOURCES)
    worker_expected = suite.get("worker_source_hashes")
    before = suite.get("source_hashes_before")
    after = suite.get("source_hashes_after")
    require(isinstance(worker_expected, dict), "suite worker_source_hashes are missing")
    require(isinstance(before, dict) and isinstance(after, dict), "suite source hashes are incomplete")
    require(set(worker_expected) == set(worker_names), "suite worker source hash inventory differs")
    require(set(before) == set(suite_names), "suite source hash inventory differs")
    require(after == before, "suite source hashes changed during the run")
    require(suite.get("sources_unchanged") is True, "suite did not attest unchanged sources")
    for name in suite_names:
        require_sha256(before.get(name), f"suite source hash {name}")
    require(worker_expected == {name: before[name] for name in worker_names},
            "suite worker hashes are not the matching subset of suite source hashes")

    current = {name: sha256_file(ROOT / name) for name in suite_names}
    require(current == before, "current repository source bytes differ from the completed suite")
    for job_name, worker in workers.items():
        require(worker.get("source_hashes") == worker_expected,
                f"{job_name} source hashes differ from the suite baseline")
        require(worker.get("source_hashes_after") == worker_expected,
                f"{job_name} source hashes changed while its worker ran")
    return {
        "passed": True,
        "suite_source_count": len(suite_names),
        "worker_source_count": len(worker_names),
        "current_source_bytes_match_recorded_hashes": True,
        "all_eight_workers_match": True,
    }


def validate_fixture(fixture: dict, fixture_path: Path, suite: dict) -> dict[str, dict]:
    require(fixture.get("status") == "frozen", "fixture is not marked frozen")
    require(fixture.get("model_revision") == PINNED_MODEL_REVISION, "fixture model revision differs")
    require(fixture.get("model_manifest_sha256") == suite.get("model_manifest_sha256"),
            "fixture model-manifest hash differs from suite")
    require(suite.get("fixture_path") == str(fixture_path), "suite fixture path differs from CLI fixture")
    fixture_digest = sha256_file(fixture_path)
    require_sha256(suite.get("fixture_sha256"), "suite fixture SHA-256")
    require(fixture_digest == suite["fixture_sha256"], "fixture bytes differ from suite record")
    require(suite.get("fixture_sha256_after") == fixture_digest, "fixture after-hash differs")
    require(suite.get("fixture_unchanged") is True, "suite did not attest unchanged fixture")

    requests = fixture.get("requests")
    require(isinstance(requests, list) and len(requests) == 4,
            "fixture must contain exactly four fixed requests")
    result = {}
    combinations = set()
    for row in requests:
        require(isinstance(row, dict), "fixture request row is not an object")
        name = row.get("name")
        token_ids = row.get("prompt_ids")
        require(isinstance(name, str) and name not in result, "fixture names must be unique strings")
        require(isinstance(token_ids, list) and token_ids, f"{name}: missing prompt token IDs")
        require(all(type(token_id) is int and token_id >= 0 for token_id in token_ids),
                f"{name}: prompt IDs must be nonnegative integers")
        tokens = len(token_ids)
        require(tokens in EXPECTED_CASE_LENGTHS, f"{name}: unexpected prompt length {tokens}")
        require(row.get("tokens") == tokens, f"{name}: fixture token count field differs")
        family = "archive" if name.lower().startswith("archive") else (
            "code" if name.lower().startswith("code") else None
        )
        require(family is not None, f"{name}: expected archive or code case")
        require(name == f"{family}-T{tokens}", f"{name}: name does not encode family and length")
        key = (family, tokens)
        require(key not in combinations, f"fixture repeats {family}/{tokens}")
        combinations.add(key)
        digest = canonical_prompt_sha256(token_ids)
        require(row.get("prompt_ids_sha256") == digest, f"{name}: prompt ID hash differs")
        result[name] = {
            "name": name,
            "family": family,
            "tokens": tokens,
            "prompt_ids_sha256": digest,
        }
    expected = {
        (family, tokens)
        for family in ("archive", "code")
        for tokens in EXPECTED_CASE_LENGTHS
    }
    require(combinations == expected, "fixture does not cover all archive/code lengths")
    ordered = sorted(result.values(), key=lambda row: (row["family"], row["tokens"]))
    require(suite.get("fixture_inputs") == ordered, "suite fixture_inputs differ from frozen fixture")
    return result


def validate_model_evidence(suite: dict, fixture: dict, workers: dict[str, dict]) -> dict:
    model_root = resolve_path(suite.get("model_path"), "model")
    require(model_root.is_dir(), f"model directory is absent: {model_root}")
    manifest_path = resolve_path(suite.get("model_manifest_path"), "model manifest")
    expected_manifest_path = (model_root / ".cache/nanokv/source-manifest.json").resolve()
    require(manifest_path == expected_manifest_path, "suite model manifest path is unexpected")
    require(manifest_path.is_file(), f"model source manifest is absent: {manifest_path}")
    manifest_digest = sha256_file(manifest_path)
    require_sha256(suite.get("model_manifest_sha256"), "suite model-manifest SHA-256")
    require(manifest_digest == suite["model_manifest_sha256"], "model manifest bytes differ from suite")
    require(suite.get("model_manifest_sha256_after") == manifest_digest,
            "model manifest after-hash differs")
    require(suite.get("model_manifest_unchanged") is True, "suite did not attest unchanged model manifest")
    require(fixture.get("model_manifest_sha256") == manifest_digest,
            "fixture model-manifest SHA-256 differs from file")

    manifest = read_json(manifest_path)
    require(manifest.get("revision") == PINNED_MODEL_REVISION, "model manifest revision differs")
    files = manifest.get("files")
    require(isinstance(files, dict) and len(files) == EXPECTED_MODEL_FILE_COUNT,
            f"expected {EXPECTED_MODEL_FILE_COUNT} manifest files")
    audited_files = []
    for relative_name, metadata in files.items():
        require(isinstance(relative_name, str) and isinstance(metadata, dict),
                "malformed model manifest file entry")
        file_path = (model_root / relative_name).resolve()
        try:
            file_path.relative_to(model_root)
        except ValueError as exc:
            raise EvidenceError(f"model manifest path escapes model root: {relative_name}") from exc
        require(file_path.is_file(), f"model file is absent: {relative_name}")
        digest = sha256_file(file_path)
        blob = None
        if metadata.get("sha256"):
            matches = digest == metadata["sha256"]
        else:
            blob = git_blob_sha1(file_path)
            matches = blob == metadata.get("blob_id")
        require(matches, f"model file differs from manifest: {relative_name}")
        audited_files.append({
            "name": relative_name,
            "sha256": digest,
            "git_blob_sha1": blob,
            "matches_manifest": True,
            "size": file_path.stat().st_size,
        })
    before = suite.get("model_audit_before")
    after = suite.get("model_audit_after")
    require(isinstance(before, dict) and before.get("status") == "passed",
            "suite model_audit_before is missing or failed")
    require(isinstance(after, dict) and after.get("status") == "passed",
            "suite model_audit_after is missing or failed")
    expected_audit = {
        "status": "passed",
        "revision": PINNED_MODEL_REVISION,
        "model_path": str(model_root),
        "files": audited_files,
    }
    require(before == expected_audit, "suite model_audit_before differs from independently hashed files")
    require(after == expected_audit, "suite model_audit_after differs from independently hashed files")
    require(suite.get("model_files_unchanged") is True, "suite did not attest unchanged model files")
    for worker_name, worker in workers.items():
        require(worker.get("model_manifest_sha256") == manifest_digest,
                f"{worker_name} model-manifest hash differs")
    return {
        "passed": True,
        "revision": PINNED_MODEL_REVISION,
        "model_manifest_sha256": manifest_digest,
        "model_files_verified": len(audited_files),
        "model_files_unchanged": True,
    }


def expected_jobs() -> list[tuple[str, str, str, int | None]]:
    return list(EXPECTED_JOBS)


def validate_jobs_and_load_workers(suite: dict, suite_path: Path) -> tuple[dict[str, dict], dict[str, dict]]:
    suite_dir = suite_path.parent.resolve(strict=True)
    require(suite.get("status") == "passed", f"suite status is {suite.get('status')!r}")
    require(suite.get("experiment") == "M23 fixed-policy native TTFT direct-store candidate",
            "unexpected suite experiment identifier")
    require(suite.get("output_dir") == str(suite_dir), "suite output_dir does not match suite location")
    require(suite.get("scope") == "two audit processes, then three fresh-process performance pairs over four fixed prompts",
            "suite workload scope differs")
    require(suite.get("orders") == [list(order) for order in EXPECTED_ORDERS],
            "suite paired order is not the registered AB/BA/AB sequence")
    jobs = suite.get("jobs")
    expected = expected_jobs()
    require(isinstance(jobs, list) and len(jobs) == len(expected), "suite must contain exactly eight jobs")
    workers = {}
    worker_files = {}
    for job, job_contract in zip(jobs, expected):
        require(isinstance(job, dict), "suite job entry is not an object")
        name, phase, arm, group = job_contract
        require(
            (job.get("name"), job.get("phase"), job.get("arm"), job.get("group")) == job_contract,
            f"suite job order/identity differs at {name}",
        )
        require(job.get("status") == "passed" and job.get("worker_status") == "passed",
                f"suite job {name} did not pass")
        path = resolve_path(job.get("worker_path"), f"{name} worker")
        path_inside(path, suite_dir, f"{name} worker")
        require(path not in worker_files.values(), f"multiple suite jobs reference {path}")
        require(path.is_file(), f"worker JSON is missing: {path}")
        recorded_hash = require_sha256(job.get("worker_sha256"), f"{name} worker SHA-256")
        actual_hash = sha256_file(path)
        require(actual_hash == recorded_hash, f"{name} worker JSON hash differs from suite")
        worker = read_json(path)
        require(worker.get("status") == "passed", f"{name} raw worker did not pass")
        require(worker.get("phase") == phase and worker.get("arm") == arm,
                f"{name} raw worker arm/phase differs")
        require(worker.get("usable_as_clean_performance") is (phase == "perf"),
                f"{name} performance-scope marker differs")
        require(job.get("candidate_provenance") == worker.get("candidate"),
                f"{name} candidate provenance copy differs")
        workers[name] = worker
        worker_files[name] = path

    log_hashes = {}
    for job in jobs:
        log_path_value = job.get("log_path")
        if isinstance(log_path_value, str) and log_path_value:
            log_path = resolve_path(log_path_value, f"{job['name']} log")
            path_inside(log_path, suite_dir, f"{job['name']} log")
            require(log_path.is_file(), f"worker log is missing: {log_path}")
            log_hashes[job["name"]] = sha256_file(log_path)
    return workers, {"worker_files": worker_files, "log_sha256": log_hashes}


def validate_candidate_provenance(worker_name: str, worker: dict, worker_hashes: dict) -> dict:
    candidate = worker.get("candidate")
    require(isinstance(candidate, dict), f"{worker_name} candidate provenance is missing")
    arm = worker["arm"]
    require(candidate.get("arm") == arm, f"{worker_name} candidate arm differs")
    require(candidate.get("runtime_count") == LAYER_COUNT, f"{worker_name} runtime count differs")
    require(candidate.get("layer_ids") == list(range(LAYER_COUNT)),
            f"{worker_name} layer IDs are incomplete or out of order")
    require(candidate.get("copy_non_blocking") is False,
            f"{worker_name} did not preserve blocking copy behavior")
    original = require_sha256(candidate.get("original_function_sha256"), f"{worker_name} original function hash")
    generated = require_sha256(candidate.get("generated_function_sha256"), f"{worker_name} generated function hash")
    helper = require_sha256(candidate.get("helper_sha256"), f"{worker_name} helper hash")
    require(helper == worker_hashes["benchmarks/native_ttft_candidate.py"],
            f"{worker_name} helper hash does not match worker source inventory")
    source_path = candidate.get("source_path")
    require(isinstance(source_path, str) and source_path.replace("\\", "/").endswith(
        "/nanovllm/sparse/m12_runtime.py"), f"{worker_name} M12 source path differs")
    if arm == "baseline":
        require(candidate.get("replacement_count") == 0, f"{worker_name} baseline has replacements")
        require(candidate.get("replacements") == [], f"{worker_name} baseline replacement list is not empty")
        require(candidate.get("baseline_untouched") is True, f"{worker_name} baseline was modified")
        require(generated == original, f"{worker_name} baseline generated hash differs")
    else:
        require(candidate.get("replacement_count") == 2, f"{worker_name} direct_store replacement count differs")
        require(candidate.get("replacements") == EXPECTED_REPLACEMENTS,
                f"{worker_name} direct_store replacements differ from registered substitutions")
        require(candidate.get("baseline_untouched") is False, f"{worker_name} direct_store was not installed")
        require(generated != original, f"{worker_name} direct_store hash did not change")
    return {
        "arm": arm,
        "runtime_count": LAYER_COUNT,
        "layer_ids": list(range(LAYER_COUNT)),
        "original_function_sha256": original,
        "generated_function_sha256": generated,
        "replacement_count": candidate["replacement_count"],
        "helper_sha256": helper,
        "copy_non_blocking": False,
    }


def requests_by_name(worker: dict, case_names: list[str], label: str) -> dict[str, dict]:
    rows = worker.get("requests")
    require(isinstance(rows, list), f"{label} request records are missing")
    mapping = {}
    for row in rows:
        require(isinstance(row, dict), f"{label} has a malformed request row")
        name = row.get("name")
        require(isinstance(name, str) and name not in mapping,
                f"{label} has an invalid or duplicate request name")
        mapping[name] = row
    require(set(mapping) == set(case_names), f"{label} request names differ from fixture")
    return mapping


def expected_prefill_schedule(tokens: int) -> list[int]:
    full, remainder = divmod(tokens, PREFILL_CHUNK)
    return [PREFILL_CHUNK] * full + ([remainder] if remainder else [])


def validate_request_identity(row: dict, fixture_row: dict, label: str) -> None:
    name = fixture_row["name"]
    require(row.get("status") == "passed", f"{label}/{name} did not pass")
    require(row.get("tokens") == fixture_row["tokens"], f"{label}/{name} token count differs")
    require(row.get("prompt_ids_sha256") == fixture_row["prompt_ids_sha256"],
            f"{label}/{name} prompt hash differs")


def validate_layer_snapshots(value, expected_count: int, label: str, *, hashes: bool = False) -> None:
    require(isinstance(value, list) and len(value) == expected_count,
            f"{label} must contain {expected_count} layer snapshots")
    for expected_layer, row in enumerate(value):
        require(isinstance(row, dict) and row.get("layer") == expected_layer,
                f"{label} layer order/inventory differs")
        if hashes:
            tensor_hashes = row.get("tensor_hashes")
            expected_names = {
                "k_history", "v_history", "representatives", "sink_k", "sink_v", "recent_k", "recent_v"
            }
            require(isinstance(tensor_hashes, dict) and set(tensor_hashes) == expected_names,
                    f"{label} tensor hash inventory differs at layer {expected_layer}")
            require(all(is_sha256(digest) for digest in tensor_hashes.values()),
                    f"{label} has an invalid tensor hash at layer {expected_layer}")


def attention_coverage(row: dict, tokens: int, label: str) -> dict:
    prefill_steps = (tokens + PREFILL_CHUNK - 1) // PREFILL_CHUNK
    attention_steps = prefill_steps + AUDIT_DECODE_STEPS
    expected = {
        (step, layer)
        for step in range(attention_steps)
        for layer in range(LAYER_COUNT)
    }
    outputs = row.get("attention_outputs")
    require(isinstance(outputs, list), f"{label} attention outputs are missing")
    observed = []
    for item in outputs:
        require(isinstance(item, dict), f"{label} has a malformed attention output")
        step, layer = item.get("step"), item.get("layer")
        shape = item.get("shape")
        require(type(step) is int and type(layer) is int, f"{label} attention key is invalid")
        require(isinstance(shape, list) and shape and all(type(dim) is int and dim > 0 for dim in shape),
                f"{label} attention shape is invalid")
        require(is_sha256(item.get("sha256")), f"{label} attention output hash is invalid")
        require(type(item.get("k_contiguous")) is bool and type(item.get("v_contiguous")) is bool,
                f"{label} input contiguity metadata is invalid")
        observed.append((step, layer))
    observed_set = set(observed)
    require(len(observed) == len(observed_set), f"{label} has duplicate attention step/layer pairs")
    require(observed_set == expected, f"{label} does not cover all expected attention step/layer pairs")

    decode = row.get("decode_selections")
    require(isinstance(decode, list) and len(decode) == LAYER_COUNT,
            f"{label} does not contain 36 decode histories")
    for layer, history in enumerate(decode):
        require(isinstance(history, list) and len(history) == AUDIT_DECODE_STEPS,
                f"{label} layer {layer} does not contain 15 decode selections")
        for step, selection in enumerate(history):
            require(isinstance(selection, list) and len(selection) == TOP_K,
                    f"{label} layer {layer} decode selection {step} is incomplete")
            require(all(type(block_id) is int for block_id in selection),
                    f"{label} layer {layer} decode selection {step} has noninteger IDs")

    scheduled = row.get("scheduled")
    prefill_schedule = expected_prefill_schedule(tokens)
    require(isinstance(scheduled, list) and len(scheduled) == attention_steps,
            f"{label} schedule must span all {attention_steps} audit steps")
    require(scheduled[:prefill_steps] == prefill_schedule,
            f"{label} prefill schedule differs from fixed chunking")
    states = row.get("states")
    require(isinstance(states, list) and len(states) == prefill_steps,
            f"{label} prefill states are incomplete")
    for index, state in enumerate(states):
        validate_layer_snapshots(state, LAYER_COUNT, f"{label} states[{index}]")
    validate_layer_snapshots(row.get("first_token_state"), LAYER_COUNT,
                             f"{label} first_token_state", hashes=True)
    validate_layer_snapshots(row.get("final_logical_state"), LAYER_COUNT,
                             f"{label} final_logical_state")
    generated_ids = row.get("generated_ids")
    require(isinstance(generated_ids, list) and len(generated_ids) == AUDIT_GENERATED_TOKENS,
            f"{label} must contain 16 generated IDs")
    require(all(type(token_id) is int for token_id in generated_ids),
            f"{label} generated IDs must be integers")
    return {
        "tokens": tokens,
        "prefill_steps": prefill_steps,
        "attention_steps": attention_steps,
        "layers_per_attention_step": LAYER_COUNT,
        "attention_pairs_expected": len(expected),
        "attention_pairs_observed": len(observed),
        "decode_layers": len(decode),
        "decode_selections_per_layer": AUDIT_DECODE_STEPS,
        "IDs_per_decode_selection": TOP_K,
    }


def without_contiguity_metadata(outputs: list[dict]) -> list[dict]:
    ignored = {"k_contiguous", "v_contiguous"}
    return [{key: value for key, value in output.items() if key not in ignored} for output in outputs]


def compare_audits(baseline: dict, candidate: dict, fixture_rows: dict[str, dict]) -> dict:
    case_names = list(fixture_rows)
    base_rows = requests_by_name(baseline, case_names, "audit-baseline")
    candidate_rows = requests_by_name(candidate, case_names, "audit-direct_store")
    coverage = {"baseline": {}, "direct_store": {}}
    cases = []
    component_names = (
        "generated_ids", "attention_outputs", "scheduled", "states", "first_token_state",
        "final_logical_state", "decode_selections",
    )
    for name in case_names:
        base = base_rows[name]
        actual = candidate_rows[name]
        validate_request_identity(base, fixture_rows[name], "audit-baseline")
        validate_request_identity(actual, fixture_rows[name], "audit-direct_store")
        coverage["baseline"][name] = attention_coverage(base, fixture_rows[name]["tokens"], f"baseline/{name}")
        coverage["direct_store"][name] = attention_coverage(actual, fixture_rows[name]["tokens"], f"direct_store/{name}")
        components = {}
        for component in component_names:
            left, right = base.get(component), actual.get(component)
            if component == "attention_outputs":
                components[component] = without_contiguity_metadata(left) == without_contiguity_metadata(right)
            else:
                components[component] = left == right
        contiguity_equal = [
            (item["k_contiguous"], item["v_contiguous"]) for item in base["attention_outputs"]
        ] == [
            (item["k_contiguous"], item["v_contiguous"]) for item in actual["attention_outputs"]
        ]
        cases.append({
            "name": name,
            "components_equal": components,
            "attention_contiguity_metadata_equal": contiguity_equal,
            "passed": all(components.values()),
        })
    require(all(case["passed"] for case in cases), "baseline/direct_store audit state or outputs differ")
    return {"passed": True, "coverage": coverage, "cases": cases}


def finite_positive(value, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise EvidenceError(f"{label} is not numeric") from exc
    require(math.isfinite(number) and number > 0, f"{label} must be finite and positive")
    return number


def geometric_mean(values: list[float]) -> float:
    require(bool(values), "cannot compute a geometric mean of no values")
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


def performance_summary(workers: dict[str, dict], fixture_rows: dict[str, dict]) -> dict:
    case_names = list(fixture_rows)
    groups = []
    pairs = []
    first_ids = {}
    for group in range(3):
        baseline_name = f"group{group}-baseline"
        candidate_name = f"group{group}-direct_store"
        require(baseline_name in workers and candidate_name in workers,
                f"performance group {group} is missing one arm")
        baseline_rows = requests_by_name(workers[baseline_name], case_names, baseline_name)
        candidate_rows = requests_by_name(workers[candidate_name], case_names, candidate_name)
        group_first_ids = {"baseline": {}, "direct_store": {}}
        for arm, rows, worker_name in (
            ("baseline", baseline_rows, baseline_name),
            ("direct_store", candidate_rows, candidate_name),
        ):
            worker = workers[worker_name]
            require(worker.get("usable_as_clean_performance") is True,
                    f"{worker_name} is not marked clean performance")
            for case_name in case_names:
                row = rows[case_name]
                fixture_row = fixture_rows[case_name]
                validate_request_identity(row, fixture_row, worker_name)
                require(row.get("measurement_instrumented") is False,
                        f"{worker_name}/{case_name} is instrumented")
                tokens = fixture_row["tokens"]
                nsteps = (tokens + PREFILL_CHUNK - 1) // PREFILL_CHUNK
                generated = row.get("generated_ids")
                require(isinstance(generated, list) and len(generated) == 1 and type(generated[0]) is int,
                        f"{worker_name}/{case_name} must contain exactly one generated ID")
                group_first_ids[arm][case_name] = generated[0]
                require(row.get("scheduled") == expected_prefill_schedule(tokens),
                        f"{worker_name}/{case_name} prefill schedule differs")
                steps = row.get("step_ms")
                require(isinstance(steps, list) and len(steps) == nsteps,
                        f"{worker_name}/{case_name} step timing count differs")
                checked_steps = [finite_positive(value, f"{worker_name}/{case_name} step") for value in steps]
                require(row.get("prefill_step_sum_ms") == sum(checked_steps),
                        f"{worker_name}/{case_name} prefill step sum is inconsistent")
                require(row.get("main_ms") == sum(checked_steps[:-1]),
                        f"{worker_name}/{case_name} main timing is inconsistent")
                require(row.get("tail_ms") == checked_steps[-1],
                        f"{worker_name}/{case_name} tail timing is inconsistent")
                for field in TIMING_FIELDS:
                    finite_positive(row.get(field), f"{worker_name}/{case_name} {field}")
            first_ids[worker_name] = dict(group_first_ids[arm])
            require(workers[worker_name].get("warmup", {}).get("tokens") == max(EXPECTED_CASE_LENGTHS),
                    f"{worker_name} did not record the largest fixture warmup")
            require(workers[worker_name].get("warmup", {}).get("measurement_instrumented") is False,
                    f"{worker_name} warmup instrumentation marker differs")
        for case_name in case_names:
            require(group_first_ids["baseline"][case_name] == group_first_ids["direct_store"][case_name],
                    f"group {group}/{case_name} first generated IDs differ")
            pair = {"group": group, "case": case_name}
            base = baseline_rows[case_name]
            candidate = candidate_rows[case_name]
            for field in TIMING_FIELDS:
                base_value = finite_positive(base[field], f"baseline {field}")
                candidate_value = finite_positive(candidate[field], f"candidate {field}")
                pair[f"{field}_baseline"] = base_value
                pair[f"{field}_direct_store"] = candidate_value
                pair[f"{field}_ratio"] = candidate_value / base_value
            base_steps = [finite_positive(value, "baseline step") for value in base["step_ms"]]
            candidate_steps = [finite_positive(value, "candidate step") for value in candidate["step_ms"]]
            pair["step_ms_baseline"] = base_steps
            pair["step_ms_direct_store"] = candidate_steps
            pair["step_ms_ratios"] = [b / a for a, b in zip(base_steps, candidate_steps)]
            pair["first_token_id"] = group_first_ids["direct_store"][case_name]
            pairs.append(pair)
        groups.append({"group": group, "first_token_ids": group_first_ids})

    metric_summaries = {}
    for metric in TIMING_FIELDS:
        metric_summaries[metric] = {
            "all_12": ratio_stats([pair[f"{metric}_ratio"] for pair in pairs]),
            "by_group": {
                str(group): ratio_stats([pair[f"{metric}_ratio"] for pair in pairs if pair["group"] == group])
                for group in range(3)
            },
            "by_case": {
                case: ratio_stats([pair[f"{metric}_ratio"] for pair in pairs if pair["case"] == case])
                for case in case_names
            },
        }
    max_steps = max(len(pair["step_ms_ratios"]) for pair in pairs)
    step_positions = {
        str(position): ratio_stats([
            pair["step_ms_ratios"][position]
            for pair in pairs if len(pair["step_ms_ratios"]) > position
        ])
        for position in range(max_steps)
    }
    all_ttft = metric_summaries["ttft_ms"]["all_12"]["geomean"]
    every_group_gains = all(
        metric_summaries["ttft_ms"]["by_group"][str(group)]["geomean"] < 1.0
        for group in range(3)
    )
    no_case_regresses = all(
        metric_summaries["ttft_ms"]["by_case"][case]["geomean"] <= 1.0
        for case in case_names
    )
    accepted = all_ttft <= 0.97 and every_group_gains and no_case_regresses
    return {
        "summary": {
            "pairs": pairs,
            "candidate_over_baseline": metric_summaries,
            "step_position_ratios": step_positions,
            "acceptance": {
                "threshold_all_12_geomean_le_0_97": all_ttft <= 0.97,
                "every_group_geomean_below_1": every_group_gains,
                "no_case_geomean_regression": no_case_regresses,
                "accepted": accepted,
            },
            "decision": "accepted" if accepted else "not_accepted",
        },
        "first_token_ids": first_ids,
        "groups": groups,
        "ttft_ratios": [pair["ttft_ms_ratio"] for pair in pairs],
    }


def validate_profile_record(profile: dict, profile_path: Path, fixture_rows: dict[str, dict],
                            suite: dict, source_hashes: dict) -> dict:
    require(profile.get("status") == "passed", "profile record did not pass")
    require(profile.get("phase") == "profile", "profile record phase differs")
    require(profile.get("arm") in ("baseline", "direct_store"), "profile record arm is invalid")
    require(profile.get("usable_as_clean_performance") is False,
            "profile record is incorrectly marked clean performance")
    require(profile.get("source_hashes") == source_hashes
            and profile.get("source_hashes_after") == source_hashes,
            "profile record source hashes differ")
    require(profile.get("fixture_sha256") == suite.get("fixture_sha256"),
            "profile record fixture hash differs")
    require(profile.get("model_manifest_sha256") == suite.get("model_manifest_sha256"),
            "profile record model-manifest hash differs")
    require(profile.get("model_revision") == PINNED_MODEL_REVISION,
            "profile record model revision differs")
    require(profile.get("candidate", {}).get("arm") == profile.get("arm"),
            "profile candidate arm differs")
    validate_candidate_provenance("profile", profile, source_hashes)
    require(profile.get("config") == suite["config_reference"],
            "profile config differs from fixed suite config")
    rows = profile.get("requests")
    require(isinstance(rows, list) and len(rows) == 1, "profile must contain one scoped request")
    row = rows[0]
    name = row.get("name") if isinstance(row, dict) else None
    require(name in fixture_rows, "profile request name is outside the fixture")
    validate_request_identity(row, fixture_rows[name], "profile")
    require(row.get("measurement_instrumented") is True,
            "profile request is not marked instrumented")
    scope = profile.get("scope")
    require(isinstance(scope, str) and "first generated token" in scope,
            "profile scope description is missing")
    # Do not extract or merge ttft_ms, step_ms, or any other profile timings.
    return {
        "path": str(profile_path),
        "sha256": sha256_file(profile_path),
        "phase": "profile",
        "arm": profile["arm"],
        "case": name,
        "scope_checked": True,
        "provenance_checked": True,
        "timings_used": False,
    }


def inspect_profile_records(suite: dict, suite_path: Path, workers: dict[str, dict],
                            worker_paths: dict, fixture_rows: dict[str, dict],
                            worker_hashes: dict) -> dict:
    suite_dir = suite_path.parent.resolve(strict=True)
    worker_path_set = {path.resolve() for path in worker_paths.values()}
    candidates = {}

    for key in ("profile_record_paths", "profiles", "profile_records"):
        entries = suite.get(key, [])
        if not entries:
            continue
        require(isinstance(entries, list), f"suite {key} must be a list")
        for entry in entries:
            if isinstance(entry, str):
                path = resolve_path(entry, "profile record")
                candidates[path] = None
            elif isinstance(entry, dict):
                embedded = entry.get("record")
                path_value = entry.get("path") or entry.get("worker_path")
                if path_value:
                    path = resolve_path(path_value, "profile record")
                    candidates[path] = None
                elif isinstance(embedded, dict) and embedded.get("phase") == "profile":
                    candidates[None] = embedded

    for path in suite_dir.glob("*.json"):
        resolved = path.resolve()
        if resolved == suite_path.resolve() or resolved in worker_path_set:
            continue
        try:
            possible = read_json(resolved)
        except EvidenceError:
            continue
        if isinstance(possible, dict) and possible.get("phase") == "profile":
            candidates[resolved] = None

    inspected = []
    for path, embedded in candidates.items():
        if path is None:
            require(isinstance(embedded, dict), "embedded profile record is malformed")
            inspected.append(validate_profile_record(
                embedded, suite_path, fixture_rows, suite, worker_hashes
            ))
            continue
        path_inside(path, suite_dir, "profile record")
        require(path.is_file(), f"referenced profile record is missing: {path}")
        profile = read_json(path)
        require(isinstance(profile, dict), f"profile record is not an object: {path}")
        inspected.append(validate_profile_record(
            profile, path, fixture_rows, suite, worker_hashes
        ))
    return {
        "records_found": len(inspected),
        "records": inspected,
        "profile_timings_used_in_performance_recomputation": False,
    }


def audit_bundle(suite_path: Path, fixture_path: Path) -> dict:
    suite = read_json(suite_path)
    fixture = read_json(fixture_path)
    require(isinstance(suite, dict), "suite record is not an object")
    require(isinstance(fixture, dict), "fixture record is not an object")
    workers, artifacts = validate_jobs_and_load_workers(suite, suite_path)
    suite_model_path = resolve_path(suite.get("model_path"), "model")
    require(suite.get("fixture_path") == str(fixture_path), "suite fixture path differs")

    source_report = validate_source_evidence(suite, workers)
    worker_hashes = suite["worker_source_hashes"]
    worker_provenance = {
        name: validate_candidate_provenance(name, worker, worker_hashes)
        for name, worker in workers.items()
    }
    original_hashes = {item["original_function_sha256"] for item in worker_provenance.values()}
    require(len(original_hashes) == 1, "workers disagree on original _store_kv function hash")
    helper_hashes = {item["helper_sha256"] for item in worker_provenance.values()}
    require(len(helper_hashes) == 1, "workers disagree on candidate helper hash")
    model_report = validate_model_evidence(suite, fixture, workers)
    fixture_rows = validate_fixture(fixture, fixture_path, suite)

    common_git_heads = {worker.get("git_head") for worker in workers.values()}
    require(len(common_git_heads) == 1 and None not in common_git_heads,
            "workers do not share one Git revision")
    configs = [worker.get("config") for worker in workers.values()]
    require(all(isinstance(config, dict) for config in configs), "worker configs are missing")
    for config in configs:
        for key, expected in EXPECTED_CONFIG.items():
            require(config.get(key) == expected, f"worker config {key} differs from fixed plan")
    require(all(config == configs[0] for config in configs), "workers do not share identical config")

    for name, worker in workers.items():
        require(worker.get("fixture_sha256") == suite.get("fixture_sha256"),
                f"{name} fixture hash differs")
        require(worker.get("fixture_path") == str(fixture_path), f"{name} fixture path differs")
        require(worker.get("model_revision") == PINNED_MODEL_REVISION, f"{name} model revision differs")
        require(worker.get("config") == configs[0], f"{name} config differs")
        require(worker.get("runtime") == workers["audit-baseline"].get("runtime"),
                f"{name} runtime versions differ from baseline audit")

    audit_baseline = workers["audit-baseline"]
    audit_candidate = workers["audit-direct_store"]
    for worker_name, worker in (("audit-baseline", audit_baseline),
                                ("audit-direct_store", audit_candidate)):
        require(worker.get("phase") == "audit", f"{worker_name} is not an audit phase")
        require(worker.get("usable_as_clean_performance") is False,
                f"{worker_name} audit is marked performance-clean")
    audit_result = compare_audits(audit_baseline, audit_candidate, fixture_rows)
    require(suite.get("audit_comparison", {}).get("passed") is True,
            "suite did not record a passed exact audit comparison")
    require(isinstance(suite.get("audit_first_token_ids"), dict),
            "suite audit first-token map is malformed")
    base_audit_rows = requests_by_name(audit_baseline, list(fixture_rows), "audit-baseline")
    expected_audit_first_ids = {name: base_audit_rows[name]["generated_ids"][0] for name in fixture_rows}
    require(suite.get("audit_first_token_ids") == expected_audit_first_ids,
            "suite audit first-token IDs differ from raw audit")

    perf_result = performance_summary(workers, fixture_rows)
    require(suite.get("performance_summary") == perf_result["summary"],
            "suite performance summary differs from independent recomputation")
    require(suite.get("performance_decision") == perf_result["summary"]["decision"],
            "suite performance decision differs from recomputed registered gate")

    profile_result = inspect_profile_records(
        suite, suite_path, workers, artifacts["worker_files"], fixture_rows, worker_hashes
    )
    return {
        "schema": 1,
        "status": "passed",
        "auditor": "stdlib-only independent M23 evidence and ratio recomputation",
        "auditor_source_sha256": sha256_file(Path(__file__)),
        "suite": {
            "path": str(suite_path),
            "sha256": sha256_file(suite_path),
            "git_head": next(iter(common_git_heads)),
            "workers_verified": len(workers),
            "worker_json_sha256": {
                name: sha256_file(path) for name, path in artifacts["worker_files"].items()
            },
            "worker_log_sha256": artifacts["log_sha256"],
        },
        "fixture": {
            "path": str(fixture_path),
            "sha256": sha256_file(fixture_path),
            "cases": list(fixture_rows.values()),
        },
        "model": model_report,
        "source_consistency": source_report,
        "candidate_provenance": worker_provenance,
        "exact_audit_comparison": audit_result,
        "performance_recomputation": {
            "pair_count": len(perf_result["summary"]["pairs"]),
            "worker_first_token_ids": perf_result["first_token_ids"],
            "groups": perf_result["groups"],
            "ttft_candidate_over_baseline_ratios": perf_result["ttft_ratios"],
            "summary": perf_result["summary"],
            "registered_gate": {
                "all_12_ttft_ratio_geomean_le_0_97": perf_result["summary"]["acceptance"][
                    "threshold_all_12_geomean_le_0_97"
                ],
                "every_group_ttft_ratio_geomean_below_1": perf_result["summary"]["acceptance"][
                    "every_group_geomean_below_1"
                ],
                "no_case_ttft_ratio_geomean_above_1": perf_result["summary"]["acceptance"][
                    "no_case_geomean_regression"
                ],
                "decision": perf_result["summary"]["decision"],
            },
        },
        "profiles": profile_result,
        "profile_timings_included_in_perf_ratios": False,
    }


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True, help="completed suite.json")
    parser.add_argument("--fixture", type=Path, required=True, help="the frozen fixture JSON")
    parser.add_argument("--output", type=Path, required=True, help="new JSON evidence report; never overwritten")
    return parser.parse_args(argv)


def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    suite_path = args.suite.expanduser()
    fixture_path = args.fixture.expanduser()
    output_path = args.output.expanduser()
    if not suite_path.is_absolute():
        suite_path = ROOT / suite_path
    if not fixture_path.is_absolute():
        fixture_path = ROOT / fixture_path
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    suite_path = suite_path.resolve()
    fixture_path = fixture_path.resolve()
    output_path = output_path.resolve()
    if output_path.suffix.lower() != ".json":
        raise ValueError("--output must name a JSON file")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite evidence report: {output_path}")

    report = None
    failure = None
    try:
        require(suite_path.is_file(), f"suite record is absent: {suite_path}")
        require(fixture_path.is_file(), f"fixture is absent: {fixture_path}")
        report = audit_bundle(suite_path, fixture_path)
    except BaseException as exc:
        failure = {"error": str(exc), "traceback": traceback.format_exc()}
        report = {
            "schema": 1,
            "status": "failed",
            "auditor": "stdlib-only independent M23 evidence and ratio recomputation",
            "suite_path": str(suite_path),
            "fixture_path": str(fixture_path),
            "failure": failure,
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(f"native TTFT evidence report: {output_path}", flush=True)
    return 0 if report.get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(run())
