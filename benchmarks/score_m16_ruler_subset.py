"""Score a RULER-derived JSONL subset with NVIDIA's unmodified metric function.

Point --constants at a pinned NVIDIA/RULER scripts/eval/synthetic/constants.py.
This avoids the legacy evaluator's NeMo dependency, but is not a full official
RULER run: task selection, data preparation, and prompt template are external.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--constants", type=Path, required=True)
    parser.add_argument("--task", choices=("niah", "variable_tracking",
                                           "common_words_extraction",
                                           "freq_words_extraction", "qa"), required=True)
    args = parser.parse_args()

    spec = importlib.util.spec_from_file_location("ruler_official_metrics", args.constants)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load official scorer: {args.constants}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    rows = [json.loads(line) for line in args.predictions.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if not rows:
        raise ValueError("no prediction rows")
    preds = [str(row["pred"]).strip() for row in rows]
    refs = [row["outputs"] for row in rows]
    if not all(isinstance(ref, list) and ref and all(isinstance(x, str) for x in ref)
               for ref in refs):
        raise ValueError("expected a nonempty list of reference strings in each outputs")
    score = module.TASKS[args.task]["metric_fn"](preds, refs)
    print(json.dumps({
        "source": str(args.predictions),
        "official_metric_source": str(args.constants),
        "task": args.task,
        "samples": len(rows),
        "score_percent": score,
        "predictions": [{"index": row.get("index"), "pred": pred, "refs": ref}
                        for row, pred, ref in zip(rows, preds, refs)],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
