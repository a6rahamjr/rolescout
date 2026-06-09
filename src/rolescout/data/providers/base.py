"""Provider interface for external job feeds."""

from __future__ import annotations

from abc import ABC, abstractmethod

from rolescout.data.contracts import JobPosting, SearchProfile


class JobProvider(ABC):
    @abstractmethod
    async def search(self, profile: SearchProfile, limit: int) -> list[JobPosting]:
        """Return normalized jobs for a profile."""
