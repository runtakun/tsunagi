"""Agent — Pipeline + LLM tool-use loop.

An Agent sends messages to an LLM, detects tool_use responses,
executes the corresponding Tool, feeds results back, and repeats
until the LLM produces a final text response or max_turns is reached.

CRITICAL DESIGN DECISIONS:
  1. The Agent does NOT wrap any LLM SDK. It defines a simple protocol
     (ChatAdapter) that the user implements for their chosen SDK.
  2. Built-in adapters for common SDKs are provided as convenience,
     but they are thin (~30 lines each) and users can easily write their own.
  3. max_turns prevents infinite loops. Default is 10.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from tsunagi.context import Context
from tsunagi.tracer import NullTracer, Tracer

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from tsunagi.tool import Tool


@dataclass
class Message:
    """Chat message used by ChatAdapter and Agent."""

    role: str  # "user", "assistant", "tool_result"
    content: str | list[dict[str, Any]]


@dataclass
class ToolCall:
    """Represents an LLM's request to invoke a tool."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class AgentEvent:
    """Event emitted during agent execution (for streaming)."""

    type: str  # "text", "tool_call", "tool_result", "error", "done"
    text: str = ""
    tool: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    result: str = ""


@runtime_checkable
class ChatAdapter(Protocol):
    """Protocol for LLM communication.

    Users implement this for their chosen SDK. The Agent only depends
    on this interface, not on any specific SDK.
    """

    async def send(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
    ) -> tuple[str | None, list[ToolCall]]:
        """Send messages to the LLM and parse the response."""


class AnthropicAdapter:
    """Adapter for the Anthropic SDK. Requires `pip install anthropic`.

    This is a convenience — users can also implement ChatAdapter directly.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-5-20250929",
        system: str = "",
        max_tokens: int = 4096,
        **client_kwargs: Any,
    ) -> None:
        self.model = model
        self.system = system
        self.max_tokens = max_tokens
        self.client_kwargs = client_kwargs
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            import importlib

            anthropic = importlib.import_module("anthropic")
            async_anthropic_cls = getattr(anthropic, "AsyncAnthropic")
            self._client = async_anthropic_cls(**self.client_kwargs)
        return self._client

    async def send(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
    ) -> tuple[str | None, list[ToolCall]]:
        client = self._get_client()

        api_messages = [{"role": m.role, "content": m.content} for m in messages]

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": api_messages,
        }
        if self.system:
            kwargs["system"] = self.system
        if tools:
            kwargs["tools"] = tools

        response = await client.messages.create(**kwargs)

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=block.input)
                )

        text = "\n".join(text_parts) if text_parts else None

        if tool_calls:
            return None, tool_calls
        return text, []


class OpenAIAdapter:
    """Adapter for the OpenAI SDK. Requires `pip install openai`."""

    def __init__(
        self,
        model: str = "gpt-4o",
        system: str = "",
        **client_kwargs: Any,
    ) -> None:
        self.model = model
        self.system = system
        self.client_kwargs = client_kwargs
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            import importlib

            openai_module = importlib.import_module("openai")
            async_openai_cls = getattr(openai_module, "AsyncOpenAI")
            self._client = async_openai_cls(**self.client_kwargs)
        return self._client

    async def send(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
    ) -> tuple[str | None, list[ToolCall]]:
        client = self._get_client()

        api_messages: list[dict[str, Any]] = []
        if self.system:
            api_messages.append({"role": "system", "content": self.system})
        api_messages.extend({"role": m.role, "content": m.content} for m in messages)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": api_messages,
        }
        if tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "parameters": t.get("input_schema", {}),
                    },
                }
                for t in tools
            ]

        response = await client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        message = choice.message

        tool_calls: list[ToolCall] = []
        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments),
                    )
                )

        if tool_calls:
            return None, tool_calls
        return message.content, []


class Agent:
    """LLM agent with tool-use capabilities."""

    def __init__(
        self,
        adapter: ChatAdapter,
        *,
        max_turns: int = 10,
        tracer: Tracer | None = None,
    ) -> None:
        self.adapter = adapter
        self.max_turns = max_turns
        self._tools: dict[str, Tool] = {}
        self._tracer: Tracer = tracer or NullTracer()

    def register(self, *tools: Tool) -> None:
        """Register one or more tools for the agent to use."""

        for t in tools:
            self._tools[t.name] = t

    async def chat(self, user_message: str) -> str:
        """Send a message and get a final text response.

        The agent loops: LLM → tool calls → LLM → ... until text response or max_turns.
        """

        messages: list[Message] = [Message(role="user", content=user_message)]
        tool_schemas = [t.to_schema() for t in self._tools.values()]
        ctx = Context(pipeline_name="agent")

        for _ in range(self.max_turns):
            text, tool_calls = await self.adapter.send(messages, tool_schemas)

            if text is not None and not tool_calls:
                return text

            for tc in tool_calls:
                tool_obj = self._tools.get(tc.name)
                if tool_obj is None:
                    result_str = json.dumps({"error": f"Unknown tool: {tc.name}"})
                else:
                    await self._tracer.on_step_start(ctx, tc.name, tc.arguments)
                    try:
                        result = await tool_obj(**tc.arguments)
                        result_str = result if isinstance(result, str) else json.dumps(result)
                        await self._tracer.on_step_end(ctx, tc.name, result_str)
                    except Exception as e:  # pragma: no cover - exercised in tests
                        result_str = json.dumps({"error": str(e)})
                        await self._tracer.on_step_error(ctx, tc.name, e)

                messages.append(
                    Message(
                        role="assistant",
                        content=[
                            {
                                "type": "tool_use",
                                "id": tc.id,
                                "name": tc.name,
                                "input": tc.arguments,
                            }
                        ],
                    ),
                )
                messages.append(
                    Message(
                        role="user",
                        content=[
                            {
                                "type": "tool_result",
                                "tool_use_id": tc.id,
                                "content": result_str,
                            }
                        ],
                    ),
                )

        raise RuntimeError(f"Agent exceeded max_turns ({self.max_turns}) without final response")

    async def stream(self, user_message: str) -> AsyncIterator[AgentEvent]:
        """Stream agent events. Yields AgentEvent objects for each action."""

        messages: list[Message] = [Message(role="user", content=user_message)]
        tool_schemas = [t.to_schema() for t in self._tools.values()]

        for _ in range(self.max_turns):
            text, tool_calls = await self.adapter.send(messages, tool_schemas)

            if text is not None and not tool_calls:
                yield AgentEvent(type="text", text=text)
                yield AgentEvent(type="done")
                return

            for tc in tool_calls:
                yield AgentEvent(type="tool_call", tool=tc.name, args=tc.arguments)

                tool_obj = self._tools.get(tc.name)
                if tool_obj is None:
                    result_str = f"Unknown tool: {tc.name}"
                else:
                    try:
                        result = await tool_obj(**tc.arguments)
                        result_str = result if isinstance(result, str) else json.dumps(result)
                    except Exception as e:  # pragma: no cover - exercised in tests
                        result_str = f"Error: {e}"
                        yield AgentEvent(type="error", text=str(e))

                yield AgentEvent(type="tool_result", tool=tc.name, result=result_str)

                messages.append(
                    Message(
                        role="assistant",
                        content=[
                            {
                                "type": "tool_use",
                                "id": tc.id,
                                "name": tc.name,
                                "input": tc.arguments,
                            }
                        ],
                    ),
                )
                messages.append(
                    Message(
                        role="user",
                        content=[
                            {
                                "type": "tool_result",
                                "tool_use_id": tc.id,
                                "content": result_str,
                            }
                        ],
                    ),
                )

        yield AgentEvent(type="error", text=f"Exceeded max_turns ({self.max_turns})")
