from __future__ import annotations

import pytest

from tsunagi import NO_RETRY, RetryConfig
from tests import conftest
from tests.conftest import fail_twice_then_succeed


def test_no_retry_default() -> None:
    assert NO_RETRY.max_attempts == 1


def test_retry_config_delay_calculation() -> None:
    cfg = RetryConfig(delay_seconds=1, backoff_factor=2, jitter=False)
    assert cfg.get_delay(0) == 1
    assert cfg.get_delay(1) == 2
    assert cfg.get_delay(2) == 4


def test_retry_max_delay_cap() -> None:
    cfg = RetryConfig(delay_seconds=5, backoff_factor=3, max_delay_seconds=2, jitter=False)
    assert cfg.get_delay(3) == 2


def test_retry_on_filter() -> None:
    cfg = RetryConfig(retry_on=lambda e: isinstance(e, ValueError))
    assert cfg.should_retry(ValueError()) is True
    assert cfg.should_retry(TypeError()) is False


@pytest.mark.asyncio
async def test_step_retry_execution() -> None:
    conftest.call_count = 0
    result = await fail_twice_then_succeed.execute(1)
    assert result == 101
    assert conftest.call_count == 3
