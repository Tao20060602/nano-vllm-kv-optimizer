import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks" / "benchmark_m16_ruler.py"
SPEC = importlib.util.spec_from_file_location("benchmark_m16_ruler", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)


def parse_args(*extra: str):
    return adapter.parse_args([
        "--input", "input.jsonl",
        "--output", "predictions.jsonl",
        "--max-tokens", "4",
        *extra,
    ])


def test_selector_static_mask_is_opt_in_and_forwarded_to_sparse_runtime():
    default_args = parse_args()
    assert not default_args.selector_static_mask
    assert adapter.build_llm_config(default_args)["sparse_selector_static_mask"] is False

    enabled_args = parse_args("--selector-static-mask")
    assert adapter.build_llm_config(enabled_args)["sparse_selector_static_mask"] is True


def test_selector_static_mask_is_rejected_for_dense_runs(capsys):
    with pytest.raises(SystemExit) as error:
        parse_args("--dense", "--selector-static-mask")

    assert error.value.code == 2
    assert "cannot be combined with --dense" in capsys.readouterr().err


def test_cpu_threads_default_and_validation(capsys):
    assert parse_args().cpu_threads == 8

    with pytest.raises(SystemExit) as error:
        parse_args("--cpu-threads", "0")

    assert error.value.code == 2
    assert "--cpu-threads must be positive" in capsys.readouterr().err


def test_quality_metadata_uses_engine_default_encode_call():
    class Tokenizer:
        def __init__(self):
            self.prompts = []

        def encode(self, prompt):
            self.prompts.append(prompt)
            return [101, 102, 103]

    tokenizer = Tokenizer()
    prompt = "<|user|>find the value<|assistant|>"
    metadata = adapter.build_quality_metadata(prompt, [8, 9, 10, 11], tokenizer, 4)

    assert tokenizer.prompts == [prompt]
    assert metadata == {
        "prompt_sha256": "cd743664a065f7fc7941899b56c66aa69f5f13895e0ae849587bbf4db26f862b",
        "input_token_length": 3,
        "output_token_ids": [8, 9, 10, 11],
        "generated_tokens": 4,
        "hit_max_tokens": True,
    }


def test_run_metadata_keeps_input_hash_config_source_and_row_audit_json_safe():
    args = parse_args("--selector-static-mask", "--limit", "1")
    llm_config = adapter.build_llm_config(args)
    source = {
        "git_head": "0123456789abcdef",
        "dirty": True,
        "dirty_paths": [" M benchmarks/benchmark_m16_ruler.py"],
        "core_file_sha256": {"nanovllm/config.py": "a" * 64},
    }
    rows = [adapter.build_row_audit(
        1,
        {"index": 7, "task": "niah", "sample_id": "case-7"},
        {
            "prompt_sha256": "b" * 64,
            "input_token_length": 2048,
            "output_token_ids": [42, 43],
            "generated_tokens": 2,
            "hit_max_tokens": False,
        },
    )]

    metadata = adapter.build_run_metadata(
        args=args,
        model="/opt/models/Qwen3-0.6B",
        llm_config=llm_config,
        input_sha256="c" * 64,
        source=source,
        run_started_at="2026-10-03T00:00:00Z",
        run_ended_at="2026-10-03T00:00:30Z",
        input_format="legacy-ruler",
        rows=rows,
    )
    round_trip = json.loads(json.dumps(metadata))

    assert round_trip["input"]["sha256"] == "c" * 64
    assert round_trip["input"]["rows_processed"] == 1
    assert round_trip["source"] == source
    assert round_trip["configuration"]["llm"]["sparse_selector_static_mask"] is True
    assert round_trip["rows"] == rows
    assert round_trip["rows"][0]["source_identity"] == {
        "task": "niah", "index": 7, "sample_id": "case-7",
    }
    assert round_trip["hit_max_tokens_count"] == 0
    assert round_trip["run_started_at"] < round_trip["run_ended_at"]


def test_metadata_sidecar_cannot_overwrite_input():
    with pytest.raises(SystemExit) as error:
        adapter.parse_args([
            "--input", "predictions.jsonl.metadata.json",
            "--output", "predictions.jsonl",
            "--max-tokens", "4",
        ])

    assert error.value.code == 2


def test_main_writes_original_prediction_format_and_quality_sidecar(monkeypatch, tmp_path, capsys):
    input_path = tmp_path / "ruler.jsonl"
    output_path = tmp_path / "predictions.jsonl"
    source_row = {"index": 11, "input": "Find the code: ", "answer_prefix": "Answer:"}
    input_path.write_text(json.dumps(source_row) + "\n", encoding="utf-8")

    class Tokenizer:
        def __init__(self):
            self.encoded_prompts = []

        def encode(self, prompt):
            self.encoded_prompts.append(prompt)
            return [1, 2, 3, 4, 5]

    instances = []

    class FakeLLM:
        def __init__(self, model, **config):
            self.model = model
            self.config = config
            self.tokenizer = Tokenizer()
            instances.append(self)

        def generate(self, prompts, sampling, use_tqdm):
            assert prompts == ["Find the code: Answer:"]
            assert sampling.max_tokens == 4
            assert sampling.temperature == 0.0
            assert sampling.ignore_eos is False
            assert use_tqdm is False
            return [{"text": " 42", "token_ids": [31, 32]}]

        def exit(self):
            pass

    class FakeSamplingParams:
        def __init__(self, max_tokens, temperature, ignore_eos):
            self.max_tokens = max_tokens
            self.temperature = temperature
            self.ignore_eos = ignore_eos

    fake_nanovllm = types.ModuleType("nanovllm")
    fake_nanovllm.LLM = FakeLLM
    fake_nanovllm.SamplingParams = FakeSamplingParams
    monkeypatch.setitem(sys.modules, "nanovllm", fake_nanovllm)
    fake_torch = types.ModuleType("torch")
    thread_counts = []
    fake_torch.set_num_threads = thread_counts.append
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(
        adapter, "source_provenance",
        lambda: {
            "git_head": "0123456789abcdef",
            "dirty": True,
            "dirty_paths": [],
            "core_file_sha256": {},
        },
    )
    monkeypatch.setattr(adapter, "resolve_model", lambda model_arg: "/fake/model")
    monkeypatch.setattr(sys, "argv", [
        "benchmark_m16_ruler.py",
        "--input", str(input_path),
        "--output", str(output_path),
        "--max-tokens", "4",
    ])

    adapter.main()
    capsys.readouterr()

    prediction = json.loads(output_path.read_text(encoding="utf-8"))
    assert prediction == {**source_row, "pred": " 42"}
    metadata = json.loads(
        adapter.metadata_path_for_output(output_path).read_text(encoding="utf-8")
    )
    assert metadata["input"]["sha256"] == adapter.sha256_file(input_path)
    assert metadata["rows"][0]["input_token_length"] == 5
    assert metadata["rows"][0]["output_token_ids"] == [31, 32]
    assert metadata["rows"][0]["generated_tokens"] == 2
    assert metadata["rows"][0]["hit_max_tokens"] is False
    assert metadata["rows"][0]["input_index"] == 11
    assert metadata["rows"][0]["source_identity"]["index"] == 11
    assert metadata["configuration"]["cpu_threads"] == 8
    assert instances[0].config["sparse_selector_static_mask"] is False
    assert thread_counts == [8]
