"""External job provider adapters."""

from rolescout.data.providers.base import JobProvider
from rolescout.data.providers.remotive import RemotiveProvider

__all__ = ["JobProvider", "RemotiveProvider"]
