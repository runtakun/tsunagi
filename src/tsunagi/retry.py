"""Retry strategies for steps.

Retry is configured per-step, not globally. SDK exceptions propagate directly
— the retry logic only catches exceptions matching the user's retry filter.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class RetryConfig:
    """Configuration for step retry behavior.

    Attributes:
        max_attempts: Total attempts including the first try. Default 1 (no retry).
        delay_seconds: Base delay between retries. Default 1.0.
        backoff_factor: Multiplier applied to delay after each retry. Default 2.0.
        jitter: If True, add random jitter to delay. Default True.
        max_delay_seconds: Cap on delay between retries. Default 60.0.
        retry_on: Optional callable that receives the exception and returns True
                  if the step should be retried. Default: retry on all exceptions.
    """

    max_attempts: int = 1
    delay_seconds: float = 1.0
    backoff_factor: float = 2.0
    jitter: bool = True
    max_delay_seconds: float = 60.0
    retry_on: Callable[[Exception], bool] | None = None

    def should_retry(self, error: Exception) -> bool:
        if self.retry_on is None:
            return True
        return self.retry_on(error)

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for the given attempt number (0-indexed)."""
        delay = self.delay_seconds * (self.backoff_factor**attempt)
        delay = min(delay, self.max_delay_seconds)
        if self.jitter:
            delay = delay * (0.5 + random.random() * 0.5)
        return delay


# Predefined retry configs for convenience
NO_RETRY = RetryConfig(max_attempts=1)
RETRY_3X = RetryConfig(max_attempts=3, delay_seconds=1.0)
RETRY_5X = RetryConfig(max_attempts=5, delay_seconds=0.5, backoff_factor=2.0)
