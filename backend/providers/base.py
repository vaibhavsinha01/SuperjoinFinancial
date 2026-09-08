"""Common interface every LLM provider must implement."""
from abc import ABC, abstractmethod


class TransientError(Exception):
    """Retryable: rate limit, timeout, 5xx, temporary network issue."""


class QuotaExhaustedError(Exception):
    """Not retryable: hard daily/monthly quota exhausted for this provider."""


class LLMProvider(ABC):
    name: str

    @abstractmethod
    def generate_json(self, prompt: str) -> dict | list:
        """Send prompt, return parsed JSON. Raise TransientError or QuotaExhaustedError on failure."""
        ...
