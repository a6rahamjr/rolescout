"""Vectorized text and structured compatibility features."""

from __future__ import annotations

import math
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from rolescout.utils.text import normalize_text, split_values, tokens


class PairFeatureExtractor:
    feature_names = (
        "word_title_similarity",
        "word_description_similarity",
        "char_title_similarity",
        "char_description_similarity",
        "title_query_coverage",
        "description_query_coverage",
        "skill_overlap",
        "location_match",
        "workplace_match",
        "experience_match",
        "job_type_match",
        "recency_score",
        "salary_present",
        "description_quality",
    )

    def __init__(
        self,
        max_word_features: int = 8000,
        max_char_features: int = 10000,
        recency_half_life_days: float = 14,
    ) -> None:
        self.word_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            max_features=max_word_features,
            sublinear_tf=True,
            strip_accents="unicode",
        )
        self.char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
            max_features=max_char_features,
            sublinear_tf=True,
        )
        self.recency_half_life_days = recency_half_life_days
        self._fitted = False

    @staticmethod
    def _profile_text(frame: pd.DataFrame) -> pd.Series:
        return (
            frame["query"].fillna("")
            + " "
            + frame["profile_skills"].fillna("").str.replace("|", " ", regex=False)
            + " "
            + frame["desired_experience"].fillna("")
        ).map(normalize_text)

    @staticmethod
    def _job_text(frame: pd.DataFrame) -> pd.Series:
        return (
            frame["title"].fillna("")
            + " "
            + frame["description"].fillna("")
            + " "
            + frame["skills"].fillna("").str.replace("|", " ", regex=False)
        ).map(normalize_text)

    def fit(self, frame: pd.DataFrame) -> PairFeatureExtractor:
        corpus = pd.concat(
            [
                self._profile_text(frame),
                frame["title"].fillna("").map(normalize_text),
                self._job_text(frame),
            ],
            ignore_index=True,
        )
        self.word_vectorizer.fit(corpus)
        self.char_vectorizer.fit(corpus)
        self._fitted = True
        return self

    @staticmethod
    def _paired_cosine(
        vectorizer: TfidfVectorizer,
        left: pd.Series,
        right: pd.Series,
    ) -> np.ndarray:
        left_matrix = vectorizer.transform(left)
        right_matrix = vectorizer.transform(right)
        return np.asarray(left_matrix.multiply(right_matrix).sum(axis=1)).ravel()

    @staticmethod
    def _coverage(query: str, value: str) -> float:
        query_tokens = tokens(query)
        if not query_tokens:
            return 0.0
        return len(query_tokens.intersection(tokens(value))) / len(query_tokens)

    @staticmethod
    def _set_overlap(left: str, right: str) -> float:
        left_values = set(split_values(left))
        right_values = set(split_values(right))
        if not left_values:
            return 0.5
        return len(left_values.intersection(right_values)) / len(left_values)

    @staticmethod
    def _location_match(desired: str, actual: str, workplace: str) -> float:
        desired_normalized = normalize_text(desired)
        actual_normalized = normalize_text(actual)
        workplace_normalized = normalize_text(workplace)
        if not desired_normalized:
            return 1.0
        if desired_normalized == "remote":
            return 1.0 if workplace_normalized == "remote" else 0.0
        if workplace_normalized == "remote":
            return 0.8
        desired_tokens = tokens(desired_normalized)
        actual_tokens = tokens(actual_normalized)
        if desired_tokens and desired_tokens.intersection(actual_tokens):
            return 1.0
        return 0.0

    @staticmethod
    def _categorical_match(desired: str, actual: str, *, allow_any: bool = True) -> float:
        desired_normalized = normalize_text(desired)
        actual_normalized = normalize_text(actual)
        if not desired_normalized or (allow_any and desired_normalized == "any"):
            return 1.0
        if not actual_normalized:
            return 0.5
        return 1.0 if desired_normalized == actual_normalized else 0.0

    @staticmethod
    def _experience_match(desired: str, actual: str) -> float:
        levels = {"intern": 0, "entry": 1, "junior": 1, "mid": 2, "senior": 3, "lead": 4}
        desired_level = levels.get(normalize_text(desired))
        actual_level = levels.get(normalize_text(actual))
        if desired_level is None:
            return 1.0
        if actual_level is None:
            return 0.5
        difference = abs(desired_level - actual_level)
        return 1.0 if difference == 0 else 0.5 if difference == 1 else 0.0

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("PairFeatureExtractor must be fitted before transform")

        profiles = self._profile_text(frame)
        titles = frame["title"].fillna("").map(normalize_text)
        descriptions = (
            frame["description"].fillna("")
            + " "
            + frame["skills"].fillna("").str.replace("|", " ", regex=False)
        ).map(normalize_text)

        posted = pd.to_datetime(frame["posted_at"], utc=True, errors="coerce")
        now = pd.Timestamp(datetime.now(UTC))
        age_days = ((now - posted).dt.total_seconds() / 86400).fillna(365).clip(lower=0)
        recency = np.exp(-math.log(2) * age_days.to_numpy() / self.recency_half_life_days)

        rows = np.column_stack(
            [
                self._paired_cosine(self.word_vectorizer, profiles, titles),
                self._paired_cosine(self.word_vectorizer, profiles, descriptions),
                self._paired_cosine(self.char_vectorizer, profiles, titles),
                self._paired_cosine(self.char_vectorizer, profiles, descriptions),
                [
                    self._coverage(query, title)
                    for query, title in zip(frame["query"], frame["title"], strict=False)
                ],
                [
                    self._coverage(query, description)
                    for query, description in zip(
                        frame["query"], frame["description"], strict=False
                    )
                ],
                [
                    self._set_overlap(profile_skills, job_skills)
                    for profile_skills, job_skills in zip(
                        frame["profile_skills"], frame["skills"], strict=False
                    )
                ],
                [
                    self._location_match(desired, actual, workplace)
                    for desired, actual, workplace in zip(
                        frame["desired_location"],
                        frame["location"],
                        frame["workplace"],
                        strict=False,
                    )
                ],
                [
                    self._categorical_match(desired, actual)
                    for desired, actual in zip(
                        frame["workplace_preference"], frame["workplace"], strict=False
                    )
                ],
                [
                    self._experience_match(desired, actual)
                    for desired, actual in zip(
                        frame["desired_experience"], frame["experience_level"], strict=False
                    )
                ],
                [
                    self._set_overlap(desired, actual)
                    for desired, actual in zip(
                        frame["desired_job_types"], frame["job_type"], strict=False
                    )
                ],
                recency,
                frame["salary"].fillna("").astype(str).str.len().gt(0).astype(float).to_numpy(),
                frame["description"]
                .fillna("")
                .map(lambda value: min(len(tokens(value)) / 60, 1.0))
                .to_numpy(),
            ]
        )
        return np.nan_to_num(rows.astype(float), nan=0.0, posinf=1.0, neginf=0.0)
