from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from tsunagi import Agent, AgentEvent, ToolCall, tool

if TYPE_CHECKING:
    from tsunagi.agent import Message


class TextAdapter:
    def __init__(self, text: str) -> None:
        self.text = text
        self.messages: list[list[Message]] = []

    async def send(self, messages, tools):  # type: ignore[no-untyped-def]
        self.messages.append(list(messages))
        return self.text, []


class ToolThenTextAdapter:
    def __init__(self, tool_call: ToolCall, final_text: str) -> None:
        self.tool_call = tool_call
        self.final_text = final_text
        self.calls = 0
        self.seen_messages: list[list[Message]] = []

    async def send(self, messages, tools):  # type: ignore[no-untyped-def]
        self.calls += 1
        self.seen_messages.append(list(messages))
        if self.calls == 1:
            return None, [self.tool_call]
        return self.final_text, []


class EndlessAdapter:
    async def send(self, messages, tools):  # type: ignore[no-untyped-def]
        return None, [ToolCall(id="1", name="noop", arguments={})]


@tool
async def adder(a: int, b: int) -> int:
    return a + b


@pytest.mark.asyncio
async def test_agent_simple_text_response() -> None:
    adapter = TextAdapter("hello")
    agent = Agent(adapter)

    result = await agent.chat("hi")
    assert result == "hello"
    assert adapter.messages[-1][0].content == "hi"


@pytest.mark.asyncio
async def test_agent_tool_call_loop() -> None:
    adapter = ToolThenTextAdapter(
        ToolCall(id="1", name="adder", arguments={"a": 2, "b": 3}),
        "done",
    )
    agent = Agent(adapter)
    agent.register(adder)

    result = await agent.chat("calc")
    assert result == "done"
    assert adapter.calls == 2


@pytest.mark.asyncio
async def test_agent_max_turns() -> None:
    agent = Agent(EndlessAdapter(), max_turns=2)
    agent.register(adder)

    with pytest.raises(RuntimeError):
        await agent.chat("loop")


@pytest.mark.asyncio
async def test_agent_unknown_tool() -> None:
    adapter = ToolThenTextAdapter(
        ToolCall(id="x", name="missing", arguments={"q": "test"}),
        "final",
    )
    agent = Agent(adapter)

    result = await agent.chat("test")
    assert result == "final"

    tool_result_message = adapter.seen_messages[-1][-1]
    assert isinstance(tool_result_message.content, list)
    content_list: list[dict[str, Any]] = tool_result_message.content
    content = content_list[0]["content"]
    assert "Unknown tool" in content


@pytest.mark.asyncio
async def test_agent_stream() -> None:
    adapter = ToolThenTextAdapter(
        ToolCall(id="1", name="adder", arguments={"a": 1, "b": 2}),
        "done",
    )
    agent = Agent(adapter)
    agent.register(adder)

    events = [event async for event in agent.stream("calc")]

    assert events[0] == AgentEvent(
        type="tool_call", tool="adder", args={"a": 1, "b": 2}
    )
    assert events[1].type == "tool_result"
    assert json.loads(events[1].result) == 3
    assert events[2] == AgentEvent(type="text", text="done")
    assert events[-1] == AgentEvent(type="done")
