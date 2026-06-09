"""Configuration loading with environment overrides."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    synthetic_output: Path
    n_queries: int
    candidates_per_query: int


@dataclass(frozen=True)
class TrainingConfig:
    test_size: float
    cv_folds: int
    c_values: tuple[float, ...]
    class_weights: tuple[str | None, ...]


@dataclass(frozen=True)
class ModelConfig:
    artifact_path: Path
    metrics_path: Path
    max_word_features: int
    max_char_features: int
    recency_half_life_days: float


@dataclass(frozen=True)
class LinkedInConfig:
    enabled: bool
    feed_url: str
    bearer_token: str = field(repr=False)
    timeout_seconds: float = 15
    cache_ttl_seconds: int = 15


@dataclass(frozen=True)
class ProviderConfig:
    remotive_url: str
    timeout_seconds: float
    cache_ttl_seconds: int
    user_agent: str
    linkedin: LinkedInConfig


@dataclass(frozen=True)
class StorageConfig:
    database_path: Path


@dataclass(frozen=True)
class ApiConfig:
    host: str
    port: int
    default_result_limit: int
    max_result_limit: int
    stream_poll_seconds: int


@dataclass(frozen=True)
class AppConfig:
    name: str
    random_seed: int
    data: DataConfig
    training: TrainingConfig
    model: ModelConfig
    provider: ProviderConfig
    storage: StorageConfig
    api: ApiConfig


def _resolve_path(value: str | Path, base_dir: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path).resolve()


def _environment_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false")


def load_config(path: str | Path | None = None) -> AppConfig:
    config_path = Path(path or os.getenv("ROLESCOUT_CONFIG", "configs/default.yaml")).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    raw: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    repo_root = config_path.parent.parent

    project = raw["project"]
    data = raw["data"]
    training = raw["training"]
    model = raw["model"]
    provider = raw["provider"]
    storage = raw["storage"]
    api = raw["api"]

    artifact_value = os.getenv("ROLESCOUT_MODEL_PATH", model["artifact_path"])
    database_value = os.getenv("ROLESCOUT_DATABASE_PATH", storage["database_path"])
    linkedin = provider.get("linkedin", {})
    linkedin_enabled = _environment_flag(
        "ROLESCOUT_LINKEDIN_ENABLED",
        bool(linkedin.get("enabled", False)),
    )
    linkedin_feed_url = os.getenv(
        "ROLESCOUT_LINKEDIN_FEED_URL",
        str(linkedin.get("feed_url", "")),
    ).strip()
    if linkedin_enabled and not linkedin_feed_url:
        raise ValueError("ROLESCOUT_LINKEDIN_FEED_URL is required when LinkedIn is enabled")

    return AppConfig(
        name=str(project["name"]),
        random_seed=int(project["random_seed"]),
        data=DataConfig(
            synthetic_output=_resolve_path(data["synthetic_output"], repo_root),
            n_queries=int(data["n_queries"]),
            candidates_per_query=int(data["candidates_per_query"]),
        ),
        training=TrainingConfig(
            test_size=float(training["test_size"]),
            cv_folds=int(training["cv_folds"]),
            c_values=tuple(float(value) for value in training["c_values"]),
            class_weights=tuple(training["class_weights"]),
        ),
        model=ModelConfig(
            artifact_path=_resolve_path(artifact_value, repo_root),
            metrics_path=_resolve_path(model["metrics_path"], repo_root),
            max_word_features=int(model["max_word_features"]),
            max_char_features=int(model["max_char_features"]),
            recency_half_life_days=float(model["recency_half_life_days"]),
        ),
        provider=ProviderConfig(
            remotive_url=str(provider["remotive_url"]),
            timeout_seconds=float(provider["timeout_seconds"]),
            cache_ttl_seconds=int(provider["cache_ttl_seconds"]),
            user_agent=str(provider["user_agent"]),
            linkedin=LinkedInConfig(
                enabled=linkedin_enabled,
                feed_url=linkedin_feed_url,
                bearer_token=os.getenv("ROLESCOUT_LINKEDIN_BEARER_TOKEN", ""),
                timeout_seconds=float(linkedin.get("timeout_seconds", 15)),
                cache_ttl_seconds=int(linkedin.get("cache_ttl_seconds", 15)),
            ),
        ),
        storage=StorageConfig(database_path=_resolve_path(database_value, repo_root)),
        api=ApiConfig(
            host=str(api["host"]),
            port=int(api["port"]),
            default_result_limit=int(api["default_result_limit"]),
            max_result_limit=int(api["max_result_limit"]),
            stream_poll_seconds=int(api.get("stream_poll_seconds", 30)),
        ),
    )
