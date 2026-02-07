from __future__ import annotations

import asyncio

import pytest

from tsunagi import Context


def test_context_creation() -> None:
    ctx1 = Context(pipeline_name="a")
    ctx2 = Context(pipeline_name="b")

    assert ctx1.run_id != ctx2.run_id
    assert ctx1.metadata == {}


@pytest.mark.asyncio
async def test_step_timing() -> None:
    ctx = Context(pipeline_name="pipe")
    timing = ctx.start_step("step1")
    await asyncio.sleep(0.01)
    timing.finish()

    assert timing.duration_ms is not None
    assert timing.duration_ms > 0


def test_context_summary() -> None:
    ctx = Context(pipeline_name="pipe")
    t1 = ctx.start_step("a")
    t1.finish()
    t2 = ctx.start_step("b")
    t2.finish()

    summary = ctx.summary()
    assert summary["pipeline"] == "pipe"
    assert len(summary["steps"]) == 2
    assert summary["steps"][0]["name"] == "a"


def test_failed_steps() -> None:
    ctx = Context(pipeline_name="pipe")
    t1 = ctx.start_step("ok")
    t1.finish()
    t2 = ctx.start_step("bad")
    t2.finish(error=ValueError("boom"))

    failed = ctx.failed_steps
    assert len(failed) == 1
    assert failed[0].step_name == "bad"
