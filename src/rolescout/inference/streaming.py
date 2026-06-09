"""Server-sent event generation for newly discovered ranked jobs."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime

from rolescout.data.contracts import SearchProfile
from rolescout.inference.service import RankingService


def _event(name: str, data: dict[str, object], event_id: str | None = None) -> str:
    lines = [f"event: {name}"]
    if event_id:
        lines.append(f"id: {event_id.replace(chr(10), '').replace(chr(13), '')}")
    lines.append(f"data: {json.dumps(data, separators=(',', ':'), ensure_ascii=True)}")
    return "\n".join(lines) + "\n\n"


class LiveJobStreamer:
    def __init__(self, ranking_service: RankingService, *, max_seen: int = 5000) -> None:
        self._ranking_service = ranking_service
        self._max_seen = max_seen

    async def events(
        self,
        profile: SearchProfile,
        *,
        limit: int,
        min_score: float,
        poll_seconds: float,
        include_existing: bool,
        max_cycles: int | None = None,
    ) -> AsyncIterator[str]:
        provider = self._ranking_service.provider
        sources = list(provider.names) if provider is not None else []
        yield _event(
            "ready",
            {
                "poll_seconds": poll_seconds,
                "sources": sources,
                "include_existing": include_existing,
            },
        )

        seen: dict[str, None] = {}
        initialized = False
        cycles = 0
        while True:
            emitted = False
            try:
                ranked = await self._ranking_service.search(
                    profile,
                    limit=limit,
                    min_score=min_score,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                yield _event(
                    "provider_error",
                    {
                        "error": type(exc).__name__,
                        "message": "Job providers are temporarily unavailable.",
                    },
                )
            else:
                current_ids = [item.job.job_id for item in ranked]
                if not initialized and not include_existing:
                    for job_id in current_ids:
                        seen[job_id] = None
                else:
                    for item in ranked:
                        job_id = item.job.job_id
                        if job_id in seen:
                            continue
                        seen[job_id] = None
                        emitted = True
                        yield _event(
                            "job",
                            {
                                "discovered_at": datetime.now(UTC).isoformat(),
                                "result": item.to_dict(),
                            },
                            event_id=job_id,
                        )
                initialized = True

                while len(seen) > self._max_seen:
                    seen.pop(next(iter(seen)))

            cycles += 1
            if max_cycles is not None and cycles >= max_cycles:
                break
            if not emitted:
                yield f": keep-alive {datetime.now(UTC).isoformat()}\n\n"
            await asyncio.sleep(poll_seconds)
