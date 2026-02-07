"""Tsunagi — SDK-first orchestration framework.

Unchained from wrappers.
"""

from tsunagi.agent import Agent, AgentEvent, AnthropicAdapter, Message, OpenAIAdapter, ToolCall
from tsunagi.context import Context, StepTiming
from tsunagi.errors import PipelineError, StepError, TsunagiError
from tsunagi.pipeline import Pipeline
from tsunagi.retry import NO_RETRY, RETRY_3X, RETRY_5X, RetryConfig
from tsunagi.step import Step, StepSequence, step
from tsunagi.tool import Tool, tool
from tsunagi.tracer import NullTracer, StdoutTracer, Tracer

__version__ = "0.1.0"

__all__ = [
    "step",
    "Step",
    "StepSequence",
    "Pipeline",
    "Agent",
    "AnthropicAdapter",
    "OpenAIAdapter",
    "Message",
    "ToolCall",
    "AgentEvent",
    "tool",
    "Tool",
    "Context",
    "StepTiming",
    "Tracer",
    "NullTracer",
    "StdoutTracer",
    "RetryConfig",
    "NO_RETRY",
    "RETRY_3X",
    "RETRY_5X",
    "TsunagiError",
    "StepError",
    "PipelineError",
]
