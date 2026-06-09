"""Typed domain contracts for search profiles and job postings."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from rolescout.utils.text import normalize_text, split_values, strip_html


def parse_datetime(value: str | datetime | None) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif value:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    else:
        parsed = datetime.now(UTC)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


@dataclass(frozen=True)
class SearchProfile:
    query: str
    location: str = ""
    skills: tuple[str, ...] = ()
    experience_level: str = ""
    workplace: str = "any"
    job_types: tuple[str, ...] = ()
    excluded_keywords: tuple[str, ...] = ()
    excluded_companies: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> SearchProfile:
        query = strip_html(str(value.get("query", ""))).strip()
        if not query:
            raise ValueError("Search profile query must not be empty")
        return cls(
            query=query,
            location=strip_html(str(value.get("location", ""))).strip(),
            skills=split_values(value.get("skills")),
            experience_level=normalize_text(str(value.get("experience_level", ""))),
            workplace=normalize_text(str(value.get("workplace", "any"))) or "any",
            job_types=split_values(value.get("job_types")),
            excluded_keywords=split_values(value.get("excluded_keywords")),
            excluded_companies=split_values(value.get("excluded_companies")),
        )

    @property
    def text(self) -> str:
        return " ".join(
            part
            for part in (
                self.query,
                " ".join(self.skills),
                self.experience_level,
            )
            if part
        )


@dataclass(frozen=True)
class JobPosting:
    job_id: str
    title: str
    company: str
    description: str
    url: str
    location: str = ""
    workplace: str = "unknown"
    experience_level: str = ""
    job_type: str = ""
    posted_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    skills: tuple[str, ...] = ()
    salary: str = ""
    source: str = "provided"

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> JobPosting:
        title = strip_html(str(value.get("title", ""))).strip()
        if not title:
            raise ValueError("Job title must not be empty")
        job_id = str(value.get("job_id") or value.get("id") or "").strip()
        url = str(value.get("url", "")).strip()
        if not job_id:
            job_id = f"{normalize_text(title)}::{normalize_text(str(value.get('company', '')))}"
        return cls(
            job_id=job_id,
            title=title,
            company=strip_html(str(value.get("company", ""))).strip(),
            description=strip_html(str(value.get("description", ""))).strip(),
            url=url,
            location=strip_html(str(value.get("location", ""))).strip(),
            workplace=normalize_text(str(value.get("workplace", "unknown"))) or "unknown",
            experience_level=normalize_text(str(value.get("experience_level", ""))),
            job_type=normalize_text(str(value.get("job_type", ""))),
            posted_at=parse_datetime(value.get("posted_at")),
            skills=split_values(value.get("skills")),
            salary=strip_html(str(value.get("salary", ""))).strip(),
            source=normalize_text(str(value.get("source", "provided"))) or "provided",
        )

    @property
    def searchable_text(self) -> str:
        return " ".join(
            part for part in (self.title, self.description, " ".join(self.skills)) if part
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["posted_at"] = self.posted_at.isoformat()
        value["skills"] = list(self.skills)
        return value


@dataclass(frozen=True)
class RankedJob:
    job: JobPosting
    score: float
    rank: int
    match_level: str
    reasons: tuple[str, ...]
    concerns: tuple[str, ...]
    contributions: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "job": self.job.to_dict(),
            "score": self.score,
            "rank": self.rank,
            "match_level": self.match_level,
            "reasons": list(self.reasons),
            "concerns": list(self.concerns),
            "contributions": self.contributions,
        }
