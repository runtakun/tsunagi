from __future__ import annotations

import pytest

from tests.conftest import add_one, always_fail, double, to_string
from tsunagi import StepError, StepSequence, step


@pytest.mark.asyncio
async def test_step_direct_call() -> None:
    assert await add_one(5) == 6


def test_step_preserves_function_name() -> None:
    assert add_one.name == "add_one"


def test_step_custom_name() -> None:
    @step(name="custom")
    async def custom(x: int) -> int:
        return x

    assert custom.name == "custom"


def test_step_requires_async() -> None:
    def not_async(x: int) -> int:
        return x

    with pytest.raises(TypeError):
        step(not_async)


@pytest.mark.asyncio
async def test_step_execute_success() -> None:
    assert await add_one.execute(5) == 6


@pytest.mark.asyncio
async def test_step_execute_failure() -> None:
    with pytest.raises(StepError) as exc:
        await always_fail.execute(1)

    err = exc.value
    assert isinstance(err.original, ValueError)
    assert err.step_name == "always_fail"
    assert err.attempts == 1


def test_step_composition() -> None:
    seq = add_one >> double
    assert isinstance(seq, StepSequence)
    assert len(seq.steps) == 2

    seq2 = add_one >> double >> to_string
    assert isinstance(seq2, StepSequence)
    assert len(seq2.steps) == 3


@pytest.mark.asyncio
async def test_step_bind() -> None:
    @step
    async def compute(x: int, y: int, z: int = 0) -> int:
        return x + y + z

    bound = compute.bind(y=10, z=5)
    assert await bound.execute(1) == 16
