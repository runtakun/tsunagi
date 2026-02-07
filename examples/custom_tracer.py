"""Custom tracer that writes JSON lines to a file."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from tsunagi import Pipeline, step


class JSONLTracer:
    """Tracer implementation that appends events to a JSONL file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    async def _write(self, record: dict[str, Any]) -> None:
        await asyncio.to_thread(
            self.path.open("a", encoding="utf-8").write, json.dumps(record) + "\n"
        )

    async def on_pipeline_start(self, ctx) -> None:  # type: ignore[no-untyped-def]
        await self._write({"event": "pipeline_start", "run_id": ctx.run_id})

    async def on_pipeline_end(self, ctx) -> None:  # type: ignore[no-untyped-def]
        await self._write({"event": "pipeline_end", "summary": ctx.summary()})

    async def on_step_start(self, ctx, step_name, input_data):  # type: ignore[no-untyped-def]
        await self._write({"event": "step_start", "step": step_name, "input": input_data})

    async def on_step_end(self, ctx, step_name, result):  # type: ignore[no-untyped-def]
        await self._write({"event": "step_end", "step": step_name, "result": result})

    async def on_step_error(self, ctx, step_name, error):  # type: ignore[no-untyped-def]
        await self._write({"event": "step_error", "step": step_name, "error": str(error)})


@step
async def greet(name: str) -> str:
    return f"Hello, {name}!"


async def main() -> None:
    tracer = JSONLTracer("./trace.jsonl")
    pipe = Pipeline("custom").use(tracer)
    await pipe.run(greet, input="Tsunagi")


if __name__ == "__main__":
    asyncio.run(main())
