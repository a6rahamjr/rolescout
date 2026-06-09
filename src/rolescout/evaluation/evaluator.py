"""Evaluate an existing model artifact against a labeled dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rolescout.data.loaders import load_dataset
from rolescout.evaluation.metrics import evaluate_scores, timed_predict, title_overlap_baseline
from rolescout.models.ranker import JobRanker
from rolescout.utils.config import load_config


def evaluate_artifact(
    model_path: str | Path,
    dataset_path: str | Path,
) -> dict[str, object]:
    model = JobRanker.load(model_path)
    frame = load_dataset(dataset_path)
    scores, latency_ms = timed_predict(model, frame)
    baseline_scores = title_overlap_baseline(frame)
    return {
        "model": evaluate_scores(frame, scores, latency_ms),
        "baseline": evaluate_scores(frame, baseline_scores),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a RoleFit model artifact")
    parser.add_argument("--config", default=None)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--model", type=Path, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    report = evaluate_artifact(
        args.model or config.model.artifact_path,
        args.dataset or config.data.synthetic_output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
