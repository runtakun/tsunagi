from __future__ import annotations

import pytest

from tests.conftest import add_one, always_fail, double, to_string
from tsunagi import NullTracer, Pipeline, PipelineError


class MockTracer:
    def __init__(self) -> None:
        self.started = False
        self.ended = False
        self.step_starts: list[str] = []
        self.step_ends: list[str] = []
        self.errors: list[Exception] = []

    async def on_pipeline_start(self, ctx) -> None:  # type: ignore[no-untyped-def]
        self.started = True

    async def on_pipeline_end(self, ctx) -> None:  # type: ignore[no-untyped-def]
        self.ended = True

    async def on_step_start(self, ctx, step_name, input_data):  # type: ignore[no-untyped-def]
        self.step_starts.append(step_name)

    async def on_step_end(self, ctx, step_name, result):  # type: ignore[no-untyped-def]
        self.step_ends.append(step_name)

    async def on_step_error(self, ctx, step_name, error):  # type: ignore[no-untyped-def]
        self.errors.append(error)


@pytest.mark.asyncio
async def test_pipeline_single_step() -> None:
    pipe = Pipeline("single")
    result = await pipe.run(add_one, input=5)
    assert result == 6


@pytest.mark.asyncio
async def test_pipeline_sequence() -> None:
    pipe = Pipeline("seq")
    result = await pipe.run(add_one >> double, input=5)
    assert result == 12


@pytest.mark.asyncio
async def test_pipeline_three_steps() -> None:
    pipe = Pipeline("three")
    result = await pipe.run(add_one >> double >> to_string, input=5)
    assert result == "12"


@pytest.mark.asyncio
async def test_pipeline_error_propagation() -> None:
    pipe = Pipeline("error")
    with pytest.raises(PipelineError) as exc:
        await pipe.run(add_one >> always_fail, input=5)

    err = exc.value
    assert err.failed_step == "always_fail"
    assert isinstance(err.original, Exception)


@pytest.mark.asyncio
async def test_pipeline_context_tracking() -> None:
    pipe = Pipeline("ctx")
    await pipe.run(add_one >> double, input=3)

    ctx = pipe.last_context
    assert ctx is not None
    assert [t.step_name for t in ctx.timings] == ["add_one", "double"]
    assert all(t.duration_ms and t.duration_ms > 0 for t in ctx.timings)


@pytest.mark.asyncio
async def test_pipeline_tracer_called() -> None:
    tracer = MockTracer()
    pipe = Pipeline("trace").use(tracer)

    await pipe.run(add_one >> double, input=2)

    assert tracer.started is True
    assert tracer.ended is True
    assert tracer.step_starts == ["add_one", "double"]
    assert tracer.step_ends == ["add_one", "double"]


@pytest.mark.asyncio
async def test_pipeline_tracer_error_called() -> None:
    tracer = MockTracer()
    pipe = Pipeline("trace_err").use(tracer)

    with pytest.raises(PipelineError):
        await pipe.run(always_fail, input=1)

    assert tracer.errors
    assert isinstance(tracer.errors[0], Exception)


@pytest.mark.asyncio
async def test_pipeline_parallel() -> None:
    pipe = Pipeline("parallel")
    results = await pipe.run_parallel([add_one, double], input=5)
    assert results == [6, 10]


@pytest.mark.asyncio
async def test_pipeline_no_tracer_overhead() -> None:
    pipe = Pipeline("no_tracer")
    result_default = await pipe.run(add_one, input=1)

    pipe_with_null = Pipeline("no_tracer_null").use(NullTracer())
    result_null = await pipe_with_null.run(add_one, input=1)

    assert result_default == result_null == 2
