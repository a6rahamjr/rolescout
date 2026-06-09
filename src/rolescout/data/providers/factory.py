"""Provider construction from application configuration."""

from rolescout.data.providers.base import JobProvider
from rolescout.data.providers.composite import CompositeProvider
from rolescout.data.providers.linkedin import LinkedInFeedProvider
from rolescout.data.providers.remotive import RemotiveProvider
from rolescout.utils.config import ProviderConfig


def build_provider(config: ProviderConfig) -> JobProvider:
    providers: list[JobProvider] = [RemotiveProvider(config)]
    if config.linkedin.enabled:
        providers.append(LinkedInFeedProvider(config.linkedin))
    return CompositeProvider(providers)
