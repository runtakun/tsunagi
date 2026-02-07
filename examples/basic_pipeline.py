"""Minimal Tsunagi pipeline example."""

from __future__ import annotations

import asyncio

from tsunagi import Pipeline, step


@step
async def add_one(x: int) -> int:
    return x + 1


@step
async def double(x: int) -> int:
    return x * 2


async def main() -> None:
    pipe = Pipeline("basic")
    result = await pipe.run(add_one >> double, input=3)
    print("Result:", result)


if __name__ == "__main__":
    asyncio.run(main())
