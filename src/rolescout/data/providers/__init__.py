"""External job provider adapters."""

from rolescout.data.providers.base import JobProvider
from rolescout.data.providers.composite import CompositeProvider
from rolescout.data.providers.factory import build_provider
from rolescout.data.providers.linkedin import LinkedInFeedProvider
from rolescout.data.providers.remotive import RemotiveProvider

__all__ = [
    "CompositeProvider",
    "JobProvider",
    "LinkedInFeedProvider",
    "RemotiveProvider",
    "build_provider",
]
