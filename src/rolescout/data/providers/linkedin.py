"""Adapter for an approved LinkedIn job feed or partner gateway."""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from typing import Any

import httpx

from rolescout.data.contracts import JobPosting, SearchProfile
from rolescout.data.providers.base import JobProvider
from rolescout.utils.config import LinkedInConfig


class LinkedInFeedProvider(JobProvider):
    name = "linkedin"

    def __init__(
        self,
        config: LinkedInConfig,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not config.feed_url:
            raise ValueError("A LinkedIn feed URL is required")
        self._config = config
        self._client = client
        self._cache: dict[str, tuple[float, list[JobPosting]]] = {}
        self._lock = asyncio.Lock()

    async def search(self, profile: SearchProfile, limit: int) -> list[JobPosting]:
        key = f"{profile.query.strip().lower()}::{profile.location.strip().lower()}"
        cached = self._cache.get(key)
        if cached and time.monotonic() - cached[0] < self._config.cache_ttl_seconds:
            return cached[1][:limit]

        async with self._lock:
            cached = self._cache.get(key)
            if cached and time.monotonic() - cached[0] < self._config.cache_ttl_seconds:
                return cached[1][:limit]

            jobs = await self._fetch(profile, limit)
            self._cache[key] = (time.monotonic(), jobs)
            return jobs[:limit]

    async def _fetch(self, profile: SearchProfile, limit: int) -> list[JobPosting]:
        owns_client = self._client is None
        client = self._client or httpx.AsyncClient(timeout=self._config.timeout_seconds)
        headers = {"Accept": "application/json"}
        if self._config.bearer_token:
            headers["Authorization"] = f"Bearer {self._config.bearer_token}"
        try:
            response = await client.get(
                self._config.feed_url,
                params={
                    "query": profile.query,
                    "location": profile.location,
                    "limit": limit,
                },
                headers=headers,
            )
            response.raise_for_status()
            return [self._normalize(item) for item in self._items(response.json())]
        finally:
            if owns_client:
                await client.aclose()

    @staticmethod
    def _items(payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if not isinstance(payload, dict):
            return []
        for key in ("jobs", "results", "elements", "data"):
            items = payload.get(key)
            if isinstance(items, list):
                return [item for item in items if isinstance(item, dict)]
        return []

    @staticmethod
    def _first(item: dict[str, Any], *keys: str, default: Any = "") -> Any:
        for key in keys:
            value = item.get(key)
            if value not in (None, "", [], {}):
                return value
        return default

    @staticmethod
    def _text(value: Any) -> str:
        if isinstance(value, dict):
            value = value.get("name") or value.get("text") or value.get("value") or ""
        return str(value)

    @staticmethod
    def _posted_at(value: Any) -> Any:
        if isinstance(value, (int, float)) or (isinstance(value, str) and value.isdigit()):
            timestamp = float(value)
            if timestamp > 10_000_000_000:
                timestamp /= 1000
            return datetime.fromtimestamp(timestamp, tz=UTC)
        return value

    @classmethod
    def _normalize(cls, item: dict[str, Any]) -> JobPosting:
        raw_id = cls._first(
            item,
            "job_id",
            "jobPostingId",
            "externalJobPostingId",
            "id",
        )
        location = cls._text(
            cls._first(item, "location", "formattedLocation", "jobLocation")
        )
        remote = cls._first(item, "remote", "isRemote", default=False)
        workplace = cls._text(
            cls._first(item, "workplace", "workplaceType", "workplace_type")
        )
        if not workplace and (remote is True or "remote" in location.lower()):
            workplace = "remote"

        return JobPosting.from_mapping(
            {
                "job_id": f"linkedin-{raw_id}" if raw_id else "",
                "title": cls._first(item, "title", "jobTitle", "job_title"),
                "company": cls._text(
                    cls._first(item, "company", "companyName", "company_name")
                ),
                "description": cls._first(
                    item,
                    "description",
                    "jobDescription",
                    "job_description",
                ),
                "url": cls._first(
                    item,
                    "url",
                    "jobUrl",
                    "job_url",
                    "applyUrl",
                    "apply_url",
                ),
                "location": location,
                "workplace": workplace or "unknown",
                "experience_level": cls._first(
                    item,
                    "experienceLevel",
                    "experience_level",
                    "seniority",
                ),
                "job_type": cls._first(
                    item,
                    "employmentStatus",
                    "employment_type",
                    "jobType",
                    "job_type",
                ),
                "posted_at": cls._posted_at(
                    cls._first(
                        item,
                        "postedAt",
                        "posted_at",
                        "listedAt",
                        "listed_at",
                        "publishedAt",
                    )
                ),
                "skills": cls._first(item, "skills", "skillNames", default=[]),
                "salary": cls._first(item, "salary", "salaryText", "salary_text"),
                "source": "linkedin",
            }
        )
