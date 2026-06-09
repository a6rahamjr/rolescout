"""Dataset loading and schema validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {
    "query_id",
    "query",
    "desired_location",
    "profile_skills",
    "desired_experience",
    "workplace_preference",
    "desired_job_types",
    "job_id",
    "title",
    "company",
    "description",
    "location",
    "workplace",
    "experience_level",
    "job_type",
    "posted_at",
    "skills",
    "salary",
    "url",
    "source",
    "label",
}


def validate_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Dataset must contain at least one row")
    if frame["query_id"].isna().any() or frame["job_id"].isna().any():
        raise ValueError("query_id and job_id must not contain null values")
    labels = set(frame["label"].astype(int).unique())
    if not labels.issubset({0, 1}) or len(labels) < 2:
        raise ValueError("label must contain both binary classes 0 and 1")
    duplicated = frame.duplicated(subset=["query_id", "job_id"])
    if duplicated.any():
        raise ValueError("Each query_id and job_id pair must be unique")

    validated = frame.copy()
    validated["label"] = validated["label"].astype(int)
    validated["posted_at"] = pd.to_datetime(validated["posted_at"], utc=True, errors="raise")
    return validated


def load_dataset(path: str | Path) -> pd.DataFrame:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    return validate_dataset(pd.read_csv(dataset_path))


def save_dataset(frame: pd.DataFrame, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = validate_dataset(frame)
    serializable.to_csv(output_path, index=False)
    return output_path
