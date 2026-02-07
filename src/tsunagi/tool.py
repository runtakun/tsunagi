"""The @tool decorator for Agent tool registration.

Tools are async functions that an Agent's LLM can invoke via tool_use.
The decorator extracts the function's name, docstring, and type hints
to auto-generate the tool schema sent to the LLM.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, get_type_hints

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


class Tool:
    """An agent-callable tool backed by an async function.

    Attributes:
        fn: The original async function.
        name: Tool name (from function name).
        description: Tool description (from docstring).
        parameters: JSON Schema-compatible parameter definitions.
    """

    def __init__(self, fn: Callable[..., Awaitable[Any]]) -> None:
        if not inspect.iscoroutinefunction(fn):
            raise TypeError(
                f"@tool requires an async function, got {type(fn).__name__}. "
                f"Hint: add 'async' before 'def {fn.__name__}'."
            )
        self.fn = fn
        self.name = fn.__name__
        self.description = (fn.__doc__ or "").strip()
        self.parameters = self._extract_parameters(fn)

    @staticmethod
    def _extract_parameters(fn: Callable[..., Any]) -> dict[str, Any]:
        """Extract JSON Schema-compatible parameters from type hints."""

        hints = get_type_hints(fn)
        sig = inspect.signature(fn)
        properties: dict[str, Any] = {}
        required: list[str] = []

        type_map: dict[type, str] = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
        }

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue

            hint = hints.get(param_name, Any)
            json_type = type_map.get(hint, "string")
            properties[param_name] = {"type": json_type}

            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    async def __call__(self, **kwargs: Any) -> Any:
        """Invoke the tool with keyword arguments."""

        return await self.fn(**kwargs)

    def to_schema(self) -> dict[str, Any]:
        """Generate the tool schema for LLM API calls.

        Returns a dict compatible with both OpenAI and Anthropic tool formats.
        The caller (Agent) adapts this to the specific API format.
        """

        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }

    def __repr__(self) -> str:
        return f"Tool({self.name})"


def tool(fn: Callable[..., Awaitable[Any]]) -> Tool:
    """Decorator to create an agent tool from an async function.

    Usage:
        @tool
        async def web_search(query: str) -> str:
            ...
    """

    return Tool(fn)
