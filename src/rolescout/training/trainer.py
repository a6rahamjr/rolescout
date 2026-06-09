"""Grouped tuning and final model training."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from rolescout.data.generator import generate_synthetic_dataset
from rolescout.data.loaders import load_dataset, save_dataset, validate_dataset
from rolescout.evaluation.metrics import evaluate_scores, timed_predict, title_overlap_baseline
from rolescout.models.ranker import JobRanker
from rolescout.utils.config import AppConfig, load_config
from rolescout.utils.logging import configure_logging
from rolescout.utils.seeding import seed_everything

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingResult:
    model: JobRanker
    metrics: dict[str, Any]
    artifact_path: Path | None = None


def _build_model(config: AppConfig, c: float, class_weight: str | None) -> JobRanker:
    return JobRanker(
        c=c,
        class_weight=class_weight,
        random_seed=config.random_seed,
        max_word_features=config.model.max_word_features,
        max_char_features=config.model.max_char_features,
        recency_half_life_days=config.model.recency_half_life_days,
    )


def _group_split(
    frame: pd.DataFrame,
    test_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_index, test_index = next(splitter.split(frame, groups=frame["query_id"].to_numpy()))
    return frame.iloc[train_index].reset_index(drop=True), frame.iloc[test_index].reset_index(
        drop=True
    )


def select_hyperparameters(frame: pd.DataFrame, config: AppConfig) -> dict[str, Any]:
    unique_groups = frame["query_id"].nunique()
    folds = min(config.training.cv_folds, unique_groups)
    if folds < 2:
        return {"c": config.training.c_values[0], "class_weight": "balanced", "cv_ndcg": 0.0}

    candidates: list[dict[str, Any]] = []
    splitter = GroupKFold(n_splits=folds)
    groups = frame["query_id"].to_numpy()
    for c in config.training.c_values:
        for class_weight in config.training.class_weights:
            fold_scores: list[float] = []
            for train_index, validation_index in splitter.split(frame, groups=groups):
                train_frame = frame.iloc[train_index].reset_index(drop=True)
                validation_frame = frame.iloc[validation_index].reset_index(drop=True)
                model = _build_model(config, c, class_weight).fit(train_frame)
                scores = model.predict_proba(validation_frame)
                fold_scores.append(evaluate_scores(validation_frame, scores)["ndcg_at_10"])
            candidates.append(
                {
                    "c": c,
                    "class_weight": class_weight,
                    "cv_ndcg": round(float(np.mean(fold_scores)), 6),
                }
            )
    return max(candidates, key=lambda item: (item["cv_ndcg"], -float(item["c"])))


def train_model(
    frame: pd.DataFrame,
    config: AppConfig,
    *,
    save_artifact: bool = True,
) -> TrainingResult:
    seed_everything(config.random_seed)
    validated = validate_dataset(frame)
    development, test = _group_split(validated, config.training.test_size, config.random_seed)
    best = select_hyperparameters(development, config)
    logger.info("selected hyperparameters: %s", best)
    model = _build_model(config, best["c"], best["class_weight"]).fit(development)

    scores, latency_ms = timed_predict(model, test)
    model_metrics = evaluate_scores(test, scores, latency_ms)
    baseline_scores = title_overlap_baseline(test)
    baseline_metrics = evaluate_scores(test, baseline_scores)
    metrics: dict[str, Any] = {
        "model": model_metrics,
        "baseline": baseline_metrics,
        "improvement": {
            "ndcg_at_10": round(model_metrics["ndcg_at_10"] - baseline_metrics["ndcg_at_10"], 6),
            "map_at_10": round(model_metrics["map_at_10"] - baseline_metrics["map_at_10"], 6),
        },
        "selected_hyperparameters": best,
        "split": {
            "development_rows": len(development),
            "test_rows": len(test),
            "development_queries": development["query_id"].nunique(),
            "test_queries": test["query_id"].nunique(),
        },
    }
    model.metadata["metrics"] = metrics

    artifact_path = None
    if save_artifact:
        artifact_path = model.save(config.model.artifact_path)
        config.model.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        config.model.metrics_path.write_text(
            json.dumps(metrics, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return TrainingResult(model=model, metrics=metrics, artifact_path=artifact_path)


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Train the RoleScout ranker")
    parser.add_argument("--config", default=None)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--regenerate", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_path = args.dataset or config.data.synthetic_output
    if args.regenerate or not dataset_path.exists():
        frame = generate_synthetic_dataset(
            n_queries=config.data.n_queries,
            candidates_per_query=config.data.candidates_per_query,
            seed=config.random_seed,
        )
        save_dataset(frame, dataset_path)
    else:
        frame = load_dataset(dataset_path)

    result = train_model(frame, config)
    print(json.dumps(result.metrics, indent=2, sort_keys=True))
    print(f"Saved model artifact to {result.artifact_path}")


if __name__ == "__main__":
    main()
