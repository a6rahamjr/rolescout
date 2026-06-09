from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from rolescout.data.generator import generate_synthetic_dataset
from rolescout.training.trainer import TrainingResult, train_model
from rolescout.utils.config import load_config


@pytest.fixture(scope="session")
def trained_result(tmp_path_factory: pytest.TempPathFactory) -> TrainingResult:
    root = tmp_path_factory.mktemp("rolescout")
    config = load_config()
    test_config = replace(
        config,
        training=replace(
            config.training,
            cv_folds=2,
            c_values=(0.5, 1.0),
            class_weights=(None, "balanced"),
        ),
        model=replace(
            config.model,
            artifact_path=Path(root) / "model.joblib",
            metrics_path=Path(root) / "metrics.json",
            max_word_features=2500,
            max_char_features=3000,
        ),
        storage=replace(config.storage, database_path=Path(root) / "alerts.db"),
    )
    frame = generate_synthetic_dataset(n_queries=80, candidates_per_query=6, seed=42)
    return train_model(frame, test_config)
