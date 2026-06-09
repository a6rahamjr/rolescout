from __future__ import annotations

from rolescout.models.ranker import JobRanker


def test_training_exports_loadable_model_and_beats_baseline(trained_result) -> None:
    metrics = trained_result.metrics

    assert trained_result.artifact_path is not None
    assert trained_result.artifact_path.exists()
    assert metrics["model"]["ndcg_at_10"] >= 0.80
    assert metrics["improvement"]["ndcg_at_10"] > 0

    loaded = JobRanker.load(trained_result.artifact_path)
    assert loaded.metadata["training_rows"] > 0
    assert len(loaded.extractor.feature_names) >= 10
