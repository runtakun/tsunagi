from __future__ import annotations

import pytest

from tsunagi import Context, NullTracer, StdoutTracer


@pytest.mark.asyncio
async def test_null_tracer_noop() -> None:
    tracer = NullTracer()
    ctx = Context(pipeline_name="noop")

    await tracer.on_pipeline_start(ctx)
    await tracer.on_step_start(ctx, "step", None)
    await tracer.on_step_end(ctx, "step", "ok")
    await tracer.on_pipeline_end(ctx)


@pytest.mark.asyncio
async def test_stdout_tracer_output(capsys) -> None:  # type: ignore[no-untyped-def]
    tracer = StdoutTracer(verbose=True)
    ctx = Context(pipeline_name="trace")

    await tracer.on_pipeline_start(ctx)
    await tracer.on_step_start(ctx, "step1", {"a": 1})
    timing = ctx.start_step("step1")
    timing.finish()
    await tracer.on_step_end(ctx, "step1", "result")
    await tracer.on_pipeline_end(ctx)

    captured = capsys.readouterr().err
    assert "Pipeline 'trace' started" in captured
    assert "step1" in captured
