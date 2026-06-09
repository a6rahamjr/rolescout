"""Explainable pointwise learning-to-rank model."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from rolescout.data.contracts import JobPosting, RankedJob, SearchProfile
from rolescout.models.features import PairFeatureExtractor
from rolescout.utils.text import normalize_text

FEATURE_REASON_LABELS = {
    "word_title_similarity": "title meaning matches your search",
    "word_description_similarity": "description matches your search intent",
    "char_title_similarity": "title closely matches your wording",
    "char_description_similarity": "job language is semantically related",
    "title_query_coverage": "title covers your requested role",
    "description_query_coverage": "description covers your requested role",
    "skill_overlap": "skills overlap with your profile",
    "location_match": "location is compatible",
    "workplace_match": "workplace preference matches",
    "experience_match": "seniority matches",
    "job_type_match": "job type matches",
    "recency_score": "posting is recent",
    "salary_present": "salary information is available",
    "description_quality": "posting contains a detailed description",
}

FEATURE_CONCERN_LABELS = {
    "word_title_similarity": "title is not a close match",
    "word_description_similarity": "description has limited overlap with your search",
    "char_title_similarity": "title wording differs from your query",
    "char_description_similarity": "job language differs from your profile",
    "title_query_coverage": "title misses part of the requested role",
    "description_query_coverage": "description misses part of the requested role",
    "skill_overlap": "few requested skills are mentioned",
    "location_match": "location may not fit",
    "workplace_match": "workplace preference may not fit",
    "experience_match": "seniority may not fit",
    "job_type_match": "job type may not fit",
    "recency_score": "posting is older",
    "salary_present": "salary is not listed",
    "description_quality": "posting has a short description",
}


def build_pair_frame(profile: SearchProfile, jobs: list[JobPosting]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "query_id": "inference",
                "query": profile.query,
                "desired_location": profile.location,
                "profile_skills": "|".join(profile.skills),
                "desired_experience": profile.experience_level,
                "workplace_preference": profile.workplace,
                "desired_job_types": "|".join(profile.job_types),
                "job_id": job.job_id,
                "title": job.title,
                "company": job.company,
                "description": job.description,
                "location": job.location,
                "workplace": job.workplace,
                "experience_level": job.experience_level,
                "job_type": job.job_type,
                "posted_at": job.posted_at,
                "skills": "|".join(job.skills),
                "salary": job.salary,
                "url": job.url,
                "source": job.source,
            }
            for job in jobs
        ]
    )


class JobRanker:
    artifact_version = 2

    def __init__(
        self,
        *,
        c: float = 1.0,
        class_weight: str | None = "balanced",
        random_seed: int = 42,
        max_word_features: int = 8000,
        max_char_features: int = 10000,
        recency_half_life_days: float = 14,
    ) -> None:
        self.c = c
        self.class_weight = class_weight
        self.random_seed = random_seed
        self.extractor = PairFeatureExtractor(
            max_word_features=max_word_features,
            max_char_features=max_char_features,
            recency_half_life_days=recency_half_life_days,
        )
        self.scaler = StandardScaler()
        self.classifier = LogisticRegression(
            C=c,
            class_weight=class_weight,
            max_iter=1000,
            random_state=random_seed,
        )
        self.metadata: dict[str, Any] = {}
        self._fitted = False

    def fit(self, frame: pd.DataFrame) -> JobRanker:
        self.extractor.fit(frame)
        features = self.extractor.transform(frame)
        scaled = self.scaler.fit_transform(features)
        self.classifier.fit(scaled, frame["label"].astype(int).to_numpy())
        self.metadata = {
            "artifact_version": self.artifact_version,
            "trained_at": datetime.now(UTC).isoformat(),
            "training_rows": len(frame),
            "feature_names": list(self.extractor.feature_names),
            "hyperparameters": {
                "c": self.c,
                "class_weight": self.class_weight,
                "random_seed": self.random_seed,
            },
        }
        self._fitted = True
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("JobRanker must be fitted before prediction")
        features = self.extractor.transform(frame)
        scaled = self.scaler.transform(features)
        return self.classifier.predict_proba(scaled)[:, 1]

    def _score_frame(
        self,
        frame: pd.DataFrame,
    ) -> tuple[np.ndarray, list[dict[str, float]]]:
        if not self._fitted:
            raise RuntimeError("JobRanker must be fitted before prediction")
        features = self.extractor.transform(frame)
        scaled = self.scaler.transform(features)
        scores = self.classifier.predict_proba(scaled)[:, 1]
        contribution_matrix = scaled * self.classifier.coef_[0]
        contributions = [
            {
                name: round(float(value), 4)
                for name, value in zip(self.extractor.feature_names, contribution_row, strict=False)
            }
            for contribution_row in contribution_matrix
        ]
        return scores, contributions

    @staticmethod
    def _is_excluded(profile: SearchProfile, job: JobPosting) -> bool:
        searchable = normalize_text(job.searchable_text)
        company = normalize_text(job.company)
        return any(term in searchable for term in profile.excluded_keywords) or any(
            term in company for term in profile.excluded_companies
        )

    @staticmethod
    def _match_level(score: float) -> str:
        if score >= 0.75:
            return "strong"
        if score >= 0.35:
            return "possible"
        return "weak"

    def rank(
        self,
        profile: SearchProfile,
        jobs: list[JobPosting],
        *,
        limit: int | None = None,
        min_score: float = 0.0,
    ) -> list[RankedJob]:
        unique_jobs: dict[str, JobPosting] = {}
        canonical_pairs: set[tuple[str, str]] = set()
        for job in jobs:
            if self._is_excluded(profile, job):
                continue
            pair = (normalize_text(job.title), normalize_text(job.company))
            if job.job_id in unique_jobs or (all(pair) and pair in canonical_pairs):
                continue
            unique_jobs[job.job_id] = job
            if all(pair):
                canonical_pairs.add(pair)

        candidates = list(unique_jobs.values())
        if not candidates:
            return []

        frame = build_pair_frame(profile, candidates)
        scores, contributions = self._score_frame(frame)
        ordering = sorted(
            (index for index in range(len(candidates)) if scores[index] >= min_score),
            key=lambda index: (
                float(scores[index]),
                candidates[index].posted_at,
                candidates[index].job_id,
            ),
            reverse=True,
        )
        if limit is not None:
            ordering = ordering[:limit]

        ranked: list[RankedJob] = []
        for rank, index in enumerate(ordering, start=1):
            positive = sorted(
                ((name, value) for name, value in contributions[index].items() if value > 0),
                key=lambda item: item[1],
                reverse=True,
            )
            reasons = tuple(FEATURE_REASON_LABELS[name] for name, _ in positive[:3])
            negative = sorted(
                ((name, value) for name, value in contributions[index].items() if value < 0),
                key=lambda item: item[1],
            )
            concerns = tuple(FEATURE_CONCERN_LABELS[name] for name, _ in negative[:2])
            ranked.append(
                RankedJob(
                    job=candidates[index],
                    score=round(float(scores[index]), 6),
                    rank=rank,
                    match_level=self._match_level(float(scores[index])),
                    reasons=reasons or ("overall profile compatibility",),
                    concerns=concerns,
                    contributions=contributions[index],
                )
            )
        return ranked

    def save(self, path: str | Path) -> Path:
        if not self._fitted:
            raise RuntimeError("Cannot save an unfitted model")
        artifact_path = Path(path)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, artifact_path)
        return artifact_path

    @classmethod
    def load(cls, path: str | Path) -> JobRanker:
        artifact_path = Path(path)
        if not artifact_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {artifact_path}")
        model = joblib.load(artifact_path)
        if not isinstance(model, cls):
            raise TypeError(f"Artifact at {artifact_path} is not a JobRanker")
        if model.artifact_version != cls.artifact_version:
            raise ValueError(
                f"Unsupported artifact version {model.artifact_version}; "
                f"expected {cls.artifact_version}"
            )
        return model

    def describe(self) -> dict[str, Any]:
        return {
            **self.metadata,
            "coefficients": {
                name: round(float(value), 6)
                for name, value in zip(
                    self.extractor.feature_names, self.classifier.coef_[0], strict=False
                )
            },
            "scaler": {
                "mean": self.scaler.mean_.tolist(),
                "scale": self.scaler.scale_.tolist(),
            },
        }
