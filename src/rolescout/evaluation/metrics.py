"""Classification and ranking metrics grouped by search query."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _dcg(labels: np.ndarray, k: int) -> float:
    values = labels[:k]
    discounts = np.log2(np.arange(2, len(values) + 2))
    return float(np.sum((2**values - 1) / discounts))


def ndcg_at_k(labels: np.ndarray, scores: np.ndarray, k: int = 10) -> float:
    order = np.argsort(scores)[::-1]
    ideal = np.sort(labels)[::-1]
    denominator = _dcg(ideal, k)
    return _dcg(labels[order], k) / denominator if denominator > 0 else 0.0


def average_precision_at_k(labels: np.ndarray, scores: np.ndarray, k: int = 10) -> float:
    order = np.argsort(scores)[::-1][:k]
    ordered_labels = labels[order]
    relevant = int(np.sum(labels))
    if relevant == 0:
        return 0.0
    precisions = [
        float(np.sum(ordered_labels[: index + 1]) / (index + 1))
        for index, value in enumerate(ordered_labels)
        if value == 1
    ]
    return float(np.sum(precisions) / min(relevant, k)) if precisions else 0.0


def evaluate_scores(
    frame: pd.DataFrame,
    scores: np.ndarray,
    latency_ms: float = 0.0,
) -> dict[str, float]:
    labels = frame["label"].astype(int).to_numpy()
    predictions = (scores >= 0.5).astype(int)
    grouped = frame.assign(_score=scores).groupby("query_id", sort=False)
    ndcg_values = [
        ndcg_at_k(group["label"].to_numpy(), group["_score"].to_numpy(), 10) for _, group in grouped
    ]
    map_values = [
        average_precision_at_k(group["label"].to_numpy(), group["_score"].to_numpy(), 10)
        for _, group in grouped
    ]
    return {
        "ndcg_at_10": round(float(np.mean(ndcg_values)), 6),
        "map_at_10": round(float(np.mean(map_values)), 6),
        "average_precision": round(float(average_precision_score(labels, scores)), 6),
        "roc_auc": round(float(roc_auc_score(labels, scores)), 6),
        "precision": round(float(precision_score(labels, predictions, zero_division=0)), 6),
        "recall": round(float(recall_score(labels, predictions, zero_division=0)), 6),
        "f1": round(float(f1_score(labels, predictions, zero_division=0)), 6),
        "latency_ms": round(float(latency_ms), 3),
        "rows": int(len(frame)),
        "queries": int(frame["query_id"].nunique()),
    }


def title_overlap_baseline(frame: pd.DataFrame) -> np.ndarray:
    def score(row: pd.Series) -> float:
        query_tokens = set(str(row["query"]).lower().split())
        title_tokens = set(str(row["title"]).lower().split())
        return len(query_tokens.intersection(title_tokens)) / max(1, len(query_tokens))

    return frame.apply(score, axis=1).to_numpy(dtype=float)


def timed_predict(model: object, frame: pd.DataFrame) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    scores = model.predict_proba(frame)
    latency_ms = (time.perf_counter() - start) * 1000
    return scores, latency_ms
