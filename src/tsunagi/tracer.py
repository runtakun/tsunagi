"""Tracer protocol and built-in StdoutTracer.

Tracers are opt-in. A pipeline with no tracer attached runs silently.
Custom tracers implement the Tracer protocol — no base class inheritance required.
"""

from __future__ import annotations

import sys
from typing import Any, Protocol, runtime_checkable

from tsunagi.context import Context


@runtime_checkable
class Tracer(Protocol):
    """Protocol for pipeline tracers.

    Implement any subset of these methods. All are optional with default no-ops.
    Using Protocol (not ABC) so any object with matching methods works — no inheritance needed.
    """

    async def on_pipeline_start(self, ctx: Context) -> None: ...
    async def on_pipeline_end(self, ctx: Context) -> None: ...
    async def on_step_start(self, ctx: Context, step_name: str, input_data: Any) -> None: ...
    async def on_step_end(self, ctx: Context, step_name: str, result: Any) -> None: ...
    async def on_step_error(self, ctx: Context, step_name: str, error: Exception) -> None: ...


class NullTracer:
    """Default tracer that does nothing. Zero overhead."""

    async def on_pipeline_start(self, ctx: Context) -> None:
        pass

    async def on_pipeline_end(self, ctx: Context) -> None:
        pass

    async def on_step_start(self, ctx: Context, step_name: str, input_data: Any) -> None:
        pass

    async def on_step_end(self, ctx: Context, step_name: str, result: Any) -> None:
        pass

    async def on_step_error(self, ctx: Context, step_name: str, error: Exception) -> None:
        pass


class StdoutTracer:
    """Simple tracer that prints to stdout. Useful for development.

    Usage:
        pipe = Pipeline("my-pipe")
        pipe.use(StdoutTracer())
    """

    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose

    async def on_pipeline_start(self, ctx: Context) -> None:
        print(f"▶ Pipeline '{ctx.pipeline_name}' started [run={ctx.run_id}]", file=sys.stderr)

    async def on_pipeline_end(self, ctx: Context) -> None:
        summary = ctx.summary()
        status = "✓" if not ctx.failed_steps else "✗"
        print(
            f"{status} Pipeline '{ctx.pipeline_name}' finished "
            f"[{summary['total_duration_ms']}ms, {len(ctx.timings)} steps]",
            file=sys.stderr,
        )

    async def on_step_start(self, ctx: Context, step_name: str, input_data: Any) -> None:
        print(f"  → {step_name}", file=sys.stderr, end="")
        if self.verbose:
            print(f" (input: {_truncate(input_data)})", file=sys.stderr, end="")
        print(file=sys.stderr)

    async def on_step_end(self, ctx: Context, step_name: str, result: Any) -> None:
        timing = ctx.timings[-1] if ctx.timings else None
        ms = f" [{timing.duration_ms:.1f}ms]" if timing and timing.duration_ms else ""
        print(f"  ✓ {step_name}{ms}", file=sys.stderr)

    async def on_step_error(self, ctx: Context, step_name: str, error: Exception) -> None:
        print(f"  ✗ {step_name} FAILED: {error}", file=sys.stderr)


def _truncate(obj: Any, max_len: int = 80) -> str:
    s = repr(obj)
    return s[:max_len] + "..." if len(s) > max_len else s
