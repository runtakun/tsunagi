"""Agent example with simple tools.

Requires an OpenAI API key (`OPENAI_API_KEY`). Install optional SDK: `pip install openai`.
"""

from __future__ import annotations

import asyncio
import math

from tsunagi import Agent, OpenAIAdapter, tool


@tool
async def web_search(query: str) -> str:
    """Mock web search returning a canned snippet."""

    return f"Results for '{query}': ..."


@tool
async def calculator(expression: str) -> str:
    """Evaluate a basic math expression."""

    try:
        return str(eval(expression, {"__builtins__": {}}, {"sqrt": math.sqrt}))
    except Exception as exc:  # pragma: no cover - example only
        return f"error: {exc}"


async def main() -> None:
    agent = Agent(
        OpenAIAdapter(model="gpt-4o-mini", system="Use tools when helpful."),
        max_turns=5,
    )
    agent.register(web_search, calculator)

    reply = await agent.chat("What is the square root of 144?")
    print(reply)


if __name__ == "__main__":
    asyncio.run(main())
