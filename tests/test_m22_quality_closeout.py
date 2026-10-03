import importlib.util
import json
from pathlib import Path
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks" / "run_m22_quality_closeout.py"
SPEC = importlib.util.spec_from_file_location("run_m22_quality_closeout", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def test_official_postprocess_is_extracted_without_importing_evaluator(tmp_path):
    source = tmp_path / "evaluate.py"
    source.write_text(
        "raise RuntimeError('the full evaluator must not execute')\n"
        "import re\n"
        "def postprocess_pred(predict_str, task_config):\n"
        "    predict_str = predict_str.strip()\n"
        "    np_pattern = re.compile(r'[\\x00-\\x1f]')\n"
        "    return np_pattern.sub('\\n', predict_str).strip()\n",
        encoding="utf-8",
    )

    postprocess = runner.load_official_postprocess(source)

    assert postprocess("  alpha\x00beta\n  ", {}) == "alpha\nbeta"


def test_dataset_count_mismatch_fails_before_any_model_job(tmp_path):
    source = tmp_path / "validation.jsonl"
    source.write_text(json.dumps({"index": 0, "input": "question", "outputs": ["answer"]}) + "\n",
                      encoding="utf-8")
    spec = runner.DatasetSpec(8192, "niah_single_1", "niah", 2, 128)

    with pytest.raises(ValueError, match="expected exactly 2"):
        runner.validate_dataset(source, spec)


def test_prompt_hash_mismatch_between_sparse_arms_fails_closeout():
    baseline = {"rows": [{"prompt_sha256": "a"}, {"prompt_sha256": "b"}]}
    m21 = {"rows": [{"prompt_sha256": "a"}, {"prompt_sha256": "different"}]}

    with pytest.raises(ValueError, match="prompt hashes differ"):
        runner.require_matched_prompts(
            {"sparse_baseline": baseline, "m21_static_mask": m21}, expected_samples=2,
        )


def test_summary_result_paths_do_not_duplicate_run_id(tmp_path, monkeypatch):
    spec = runner.DatasetSpec(32768, "niah_single_1", "niah", 1, 128)
    monkeypatch.setattr(runner, "DATASETS", (spec,))
    run_id = "fixed-run"
    result_dir = tmp_path / run_id
    plan = {"results_dir": str(result_dir), "model": "fixed-model"}
    result = {
        "rows": [{"prompt_sha256": "same", "output_token_ids": [1], "prediction": "answer"}],
        "adapter_source": {"core_file_sha256": {"runtime": "same-source"}},
        "metric": {"score_percent": 100.0},
        "hit_max_tokens_count": 0,
    }
    results = {(32768, spec.task_name, arm): result for arm in runner.arms_for(spec)}
    summary = runner.build_summary(plan, "plan-hash", results, run_id)
    for arm, values in summary["tasks"][0]["arms"].items():
        assert Path(values["result_path"]) == result_dir / f"32768_niah_single_1_{arm}.json"
