from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import httpx

from rolescout.data.contracts import JobPosting, SearchProfile
from rolescout.data.providers.base import JobProvider
from rolescout.data.providers.composite import CompositeProvider
from rolescout.data.providers.linkedin import LinkedInFeedProvider
from rolescout.utils.config import LinkedInConfig


def _job(
    job_id: str,
    title: str,
    company: str,
    source: str,
    *,
    posted_at: datetime,
) -> JobPosting:
    return JobPosting(
        job_id=job_id,
        title=title,
        company=company,
        description="Build reliable Python services.",
        url=f"https://example.com/{job_id}",
        posted_at=posted_at,
        source=source,
    )


class StaticProvider(JobProvider):
    def __init__(self, provider_name: str, jobs: list[JobPosting]) -> None:
        self._name = provider_name
        self._jobs = jobs

    @property
    def name(self) -> str:
        return self._name

    async def search(self, profile: SearchProfile, limit: int) -> list[JobPosting]:
        return self._jobs[:limit]


class FailingProvider(JobProvider):
    name = "failing"

    async def search(self, profile: SearchProfile, limit: int) -> list[JobPosting]:
        raise RuntimeError("provider unavailable")


def test_linkedin_feed_provider_normalizes_and_caches() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert request.headers["authorization"] == "Bearer test-value"
        assert request.url.params["query"] == "backend engineer"
        assert request.url.params["location"] == "Berlin"
        return httpx.Response(
            200,
            json={
                "jobs": [
                    {
                        "jobPostingId": "123",
                        "jobTitle": "Senior Backend Engineer",
                        "company": {"name": "Northstar Labs"},
                        "jobDescription": "Build APIs with Python and PostgreSQL.",
                        "jobUrl": "https://www.linkedin.com/jobs/view/123",
                        "formattedLocation": "Berlin (Remote)",
                        "isRemote": True,
                        "employmentStatus": "FULL_TIME",
                        "listedAt": 1_710_000_000_000,
                        "skillNames": ["Python", "PostgreSQL"],
                    }
                ]
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    config = LinkedInConfig(
        enabled=True,
        feed_url="https://feed.example.test/linkedin/jobs",
        bearer_token="test-value",
        cache_ttl_seconds=60,
    )
    provider = LinkedInFeedProvider(config, client=client)
    profile = SearchProfile(query="backend engineer", location="Berlin")

    async def run() -> tuple[list[JobPosting], list[JobPosting]]:
        try:
            return await provider.search(profile, 10), await provider.search(profile, 10)
        finally:
            await client.aclose()

    first, second = asyncio.run(run())
    assert calls == 1
    assert first == second
    assert first[0].job_id == "linkedin-123"
    assert first[0].company == "Northstar Labs"
    assert first[0].workplace == "remote"
    assert first[0].source == "linkedin"
    assert first[0].posted_at.year == 2024


def test_composite_provider_merges_deduplicates_and_tolerates_failure() -> None:
    now = datetime.now(UTC)
    remotive = StaticProvider(
        "remotive",
        [
            _job(
                "remotive-1",
                "Backend Engineer",
                "Northstar Labs",
                "remotive",
                posted_at=now - timedelta(hours=2),
            ),
            _job(
                "remotive-2",
                "Data Engineer",
                "Bluebird",
                "remotive",
                posted_at=now,
            ),
        ],
    )
    linkedin = StaticProvider(
        "linkedin",
        [
            _job(
                "linkedin-1",
                "Backend Engineer",
                "Northstar Labs",
                "linkedin",
                posted_at=now - timedelta(hours=1),
            )
        ],
    )
    provider = CompositeProvider([remotive, linkedin, FailingProvider()])

    jobs = asyncio.run(provider.search(SearchProfile(query="engineer"), 10))

    assert provider.names == ("remotive", "linkedin", "failing")
    assert [job.job_id for job in jobs] == ["remotive-2", "linkedin-1"]
