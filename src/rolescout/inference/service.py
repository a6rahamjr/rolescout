"""Application-facing inference orchestration."""

from __future__ import annotations

from rolescout.data.contracts import JobPosting, RankedJob, SearchProfile
from rolescout.data.providers.base import JobProvider
from rolescout.models.ranker import JobRanker


class RankingService:
    def __init__(self, ranker: JobRanker, provider: JobProvider | None = None) -> None:
        self.ranker = ranker
        self.provider = provider

    def rank(
        self,
        profile: SearchProfile,
        jobs: list[JobPosting],
        *,
        limit: int | None = None,
        min_score: float = 0.0,
    ) -> list[RankedJob]:
        return self.ranker.rank(profile, jobs, limit=limit, min_score=min_score)

    async def search(
        self,
        profile: SearchProfile,
        *,
        limit: int,
        min_score: float = 0.0,
    ) -> list[RankedJob]:
        if self.provider is None:
            raise RuntimeError("No job provider is configured")
        jobs = await self.provider.search(profile, limit=max(limit * 3, limit))
        return self.rank(profile, jobs, limit=limit, min_score=min_score)
