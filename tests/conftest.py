"""Shared fixtures for Tsunagi tests."""

from __future__ import annotations

import pytest

from tsunagi import RetryConfig, step


@step
async def add_one(x: int) -> int:
    return x + 1


@step
async def double(x: int) -> int:
    return x * 2


@step
async def to_string(x: int) -> str:
    return str(x)


@step
async def always_fail(x: int) -> int:
    raise ValueError("intentional failure")


call_count: int = 0


@step(retry=RetryConfig(max_attempts=3, delay_seconds=0.01))
async def fail_twice_then_succeed(x: int) -> int:
    global call_count
    call_count += 1
    if call_count < 3:
        raise ValueError(f"attempt {call_count}")
    return x + 100


@pytest.fixture(autouse=True)
def reset_call_count() -> None:
    """Reset call count between tests."""

    global call_count
    call_count = 0
