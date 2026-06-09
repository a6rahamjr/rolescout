"""Remotive public API adapter with in-process response caching."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx

from rolescout.data.contracts import JobPosting, SearchProfile
from rolescout.data.providers.base import JobProvider
from rolescout.utils.config import ProviderConfig


class RemotiveProvider(JobProvider):
    name = "remotive"

    def __init__(self, config: ProviderConfig, client: httpx.AsyncClient | None = None) -> None:
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
        client = self._client or httpx.AsyncClient(
            timeout=self._config.timeout_seconds,
            headers={"User-Agent": self._config.user_agent},
        )
        try:
            response = await client.get(
                self._config.remotive_url,
                params={"search": profile.query, "limit": limit},
            )
            response.raise_for_status()
            payload: dict[str, Any] = response.json()
            return [self._normalize(item) for item in payload.get("jobs", [])]
        finally:
            if owns_client:
                await client.aclose()

    @staticmethod
    def _normalize(item: dict[str, Any]) -> JobPosting:
        location = str(item.get("candidate_required_location", "Remote"))
        return JobPosting.from_mapping(
            {
                "job_id": f"remotive-{item['id']}",
                "title": item.get("title", ""),
                "company": item.get("company_name", ""),
                "description": item.get("description", ""),
                "url": item.get("url", ""),
                "location": location,
                "workplace": "remote",
                "experience_level": "",
                "job_type": str(item.get("job_type", "")).replace("_", " "),
                "posted_at": item.get("publication_date"),
                "skills": [],
                "salary": item.get("salary", ""),
                "source": "remotive",
            }
        )
