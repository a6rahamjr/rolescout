from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from rolescout.data.contracts import JobPosting, RankedJob, SearchProfile
from rolescout.data.providers.base import JobProvider
from rolescout.inference.streaming import LiveJobStreamer


class StreamProvider(JobProvider):
    name = "test-feed"

    async def search(self, profile: SearchProfile, limit: int) -> list[JobPosting]:
        return []


def _ranked(job_id: str) -> RankedJob:
    return RankedJob(
        job=JobPosting(
            job_id=job_id,
            title=f"Python Engineer {job_id}",
            company="Northstar Labs",
            description="Build production services.",
            url=f"https://example.com/{job_id}",
            posted_at=datetime.now(UTC),
            source="test-feed",
        ),
        score=0.9,
        rank=1,
        match_level="strong",
        reasons=("title meaning matches your search",),
        concerns=(),
        contributions={},
    )


class SequenceRankingService:
    def __init__(self) -> None:
        self.provider = StreamProvider()
        self._calls = 0

    async def search(
        self,
        profile: SearchProfile,
        *,
        limit: int,
        min_score: float,
    ) -> list[RankedJob]:
        self._calls += 1
        if self._calls == 1:
            return [_ranked("existing")]
        return [_ranked("new"), _ranked("existing")]


def test_stream_emits_only_jobs_discovered_after_connection() -> None:
    service = SequenceRankingService()
    streamer = LiveJobStreamer(service)  # type: ignore[arg-type]

    async def collect() -> list[str]:
        return [
            chunk
            async for chunk in streamer.events(
                SearchProfile(query="python engineer"),
                limit=10,
                min_score=0.5,
                poll_seconds=0,
                include_existing=False,
                max_cycles=2,
            )
        ]

    chunks = asyncio.run(collect())
    output = "".join(chunks)
    assert "event: ready" in output
    assert '"sources":["test-feed"]' in output
    assert output.count("event: job") == 1
    assert "id: new" in output
    assert "id: existing" not in output
