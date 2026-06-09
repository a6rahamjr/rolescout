"""Concurrent provider aggregation with deterministic deduplication."""

from __future__ import annotations

import asyncio
import logging

from rolescout.data.contracts import JobPosting, SearchProfile
from rolescout.data.providers.base import JobProvider
from rolescout.utils.text import normalize_text

logger = logging.getLogger(__name__)


class CompositeProvider(JobProvider):
    name = "composite"

    def __init__(self, providers: list[JobProvider]) -> None:
        if not providers:
            raise ValueError("At least one job provider is required")
        self._providers = tuple(providers)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(name for provider in self._providers for name in provider.names))

    async def search(self, profile: SearchProfile, limit: int) -> list[JobPosting]:
        results = await asyncio.gather(
            *(provider.search(profile, limit) for provider in self._providers),
            return_exceptions=True,
        )
        jobs: list[JobPosting] = []
        errors: list[BaseException] = []
        for provider, result in zip(self._providers, results, strict=True):
            if isinstance(result, BaseException):
                logger.warning("Job provider %s failed: %s", provider.name, type(result).__name__)
                errors.append(result)
            else:
                jobs.extend(result)

        if not jobs and len(errors) == len(self._providers):
            raise errors[0]
        return self._deduplicate(jobs)[:limit]

    @staticmethod
    def _deduplicate(jobs: list[JobPosting]) -> list[JobPosting]:
        ordered = sorted(jobs, key=lambda job: (job.posted_at, job.job_id), reverse=True)
        seen_ids: set[str] = set()
        seen_roles: set[tuple[str, str]] = set()
        unique: list[JobPosting] = []
        for job in ordered:
            role = (normalize_text(job.title), normalize_text(job.company))
            if job.job_id in seen_ids or (all(role) and role in seen_roles):
                continue
            seen_ids.add(job.job_id)
            if all(role):
                seen_roles.add(role)
            unique.append(job)
        return unique
