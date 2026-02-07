from __future__ import annotations

import pytest

from tsunagi import Tool, tool


def test_tool_creation() -> None:
    @tool
    async def sample() -> str:
        """Sample tool."""

        return "ok"

    assert isinstance(sample, Tool)
    assert sample.name == "sample"
    assert sample.description == "Sample tool."


def test_tool_parameter_extraction() -> None:
    @tool
    async def search(query: str, limit: int = 5) -> str:
        return f"{query}:{limit}"

    schema = search.parameters
    assert schema["required"] == ["query"]
    assert schema["properties"]["query"]["type"] == "string"
    assert schema["properties"]["limit"]["type"] == "integer"


def test_tool_schema_generation() -> None:
    @tool
    async def sample() -> str:
        """Another sample."""

        return "ok"

    schema = sample.to_schema()
    assert schema["name"] == "sample"
    assert schema["description"] == "Another sample."
    assert "input_schema" in schema


@pytest.mark.asyncio
async def test_tool_invocation() -> None:
    @tool
    async def echo(query: str) -> str:
        return query.upper()

    assert await echo(query="hi") == "HI"


def test_tool_requires_async() -> None:
    def sync_tool() -> str:
        return "nope"

    with pytest.raises(TypeError):
        tool(sync_tool)  # type: ignore[arg-type]
