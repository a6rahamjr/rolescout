"""Application-facing inference orchestration."""

from __future__ import annotations

from rolescout.data.contracts import JobPosting, RankedJob, SearchProfile
from rolescout.data.providers.base import JobProvider
from rolescout.models.ranker import JobRanker
from rolescout.utils.text import tokens

_SEARCH_STOPWORDS = {"and", "for", "in", "of", "on", "the", "to"}


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
        relevant = [job for job in jobs if self._matches_search_intent(profile, job)]
        return self.rank(profile, relevant, limit=limit, min_score=min_score)

    @staticmethod
    def _matches_search_intent(profile: SearchProfile, job: JobPosting) -> bool:
        raw_terms = tokens(profile.query)
        intent_terms = raw_terms - _SEARCH_STOPWORDS or raw_terms
        if not intent_terms:
            return True

        title_terms = tokens(job.title)
        posting_terms = tokens(job.searchable_text)
        title_overlap = intent_terms & title_terms
        total_overlap = intent_terms & posting_terms
        required_matches = 1 if len(intent_terms) == 1 else 2
        return bool(title_overlap) and len(total_overlap) >= required_matches
