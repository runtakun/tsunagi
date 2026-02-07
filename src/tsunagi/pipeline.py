"""Pipeline — the core orchestration engine.

A Pipeline runs a sequence of Steps, passing each step's output as
the next step's input. It manages Context, Tracing, and error propagation.

CRITICAL DESIGN DECISIONS:
  1. SDK exceptions are NEVER caught-and-rewrapped silently.
     StepError wraps the original exception with `.original` attribute.
  2. Tracing is opt-in. A pipeline with no tracer has zero tracing overhead.
  3. The pipeline does NOT know about any specific SDK (OpenAI, Anthropic, etc.)
"""

from __future__ import annotations

import asyncio
from typing import Any

from tsunagi.context import Context
from tsunagi.errors import PipelineError
from tsunagi.step import Step, StepSequence
from tsunagi.tracer import NullTracer, Tracer


class Pipeline:
    """Orchestrates a sequence of Steps.

    Usage:
        pipe = Pipeline("rag")
        pipe.use(StdoutTracer())  # opt-in

        result = await pipe.run(embed >> search >> generate, input="query")

        # Or run individual steps with pipeline features (retry, tracing):
        result = await pipe.run(embed, input="query")
    """

    def __init__(self, name: str = "default") -> None:
        self.name = name
        self._tracer: Tracer = NullTracer()
        self._last_context: Context | None = None

    @property
    def last_context(self) -> Context | None:
        """Most recent execution context, if any."""

        return self._last_context

    def use(self, tracer: Tracer) -> Pipeline:
        """Attach a tracer. Returns self for chaining.

        Args:
            tracer: Any object implementing the Tracer protocol.
        """

        self._tracer = tracer
        return self

    async def run(
        self,
        steps: Step | StepSequence,
        *,
        input: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        """Execute a step or step sequence.

        Args:
            steps: A single Step or a StepSequence (created via >> operator).
            input: Initial input passed to the first step.
            metadata: Optional metadata dict attached to the Context.

        Returns:
            The output of the last step in the sequence.

        Raises:
            PipelineError: If any step fails (wraps the original exception).
        """

        if isinstance(steps, Step):
            step_list = [steps]
        elif isinstance(steps, StepSequence):
            step_list = steps.steps
        else:
            raise TypeError(f"Expected Step or StepSequence, got {type(steps).__name__}")

        ctx = Context(pipeline_name=self.name, metadata=metadata or {})
        self._last_context = ctx

        await self._tracer.on_pipeline_start(ctx)

        current = input

        try:
            for s in step_list:
                timing = ctx.start_step(s.name)
                await self._tracer.on_step_start(ctx, s.name, current)

                try:
                    current = await s.execute(current)
                    timing.finish()
                    await self._tracer.on_step_end(ctx, s.name, current)

                except Exception as e:  # pragma: no cover - tested via behavior
                    timing.finish(error=e)
                    await self._tracer.on_step_error(ctx, s.name, e)
                    raise PipelineError(s.name, e) from e

        finally:
            await self._tracer.on_pipeline_end(ctx)

        return current

    async def run_parallel(
        self,
        steps: list[Step],
        *,
        inputs: list[Any] | None = None,
        input: Any = None,
    ) -> list[Any]:
        """Execute multiple steps in parallel.

        Args:
            steps: List of Steps to run concurrently.
            inputs: Per-step inputs (must match len(steps)). Mutually exclusive with input.
            input: Single input passed to all steps. Mutually exclusive with inputs.

        Returns:
            List of results in the same order as steps.

        Raises:
            PipelineError: If any step fails.
        """

        if inputs is not None and input is not None:
            raise ValueError("Provide either 'inputs' or 'input', not both")

        if inputs is not None:
            if len(inputs) != len(steps):
                raise ValueError(f"inputs length ({len(inputs)}) != steps length ({len(steps)})")
            step_inputs = inputs
        else:
            step_inputs = [input] * len(steps)

        try:
            tasks = [s.execute(inp) for s, inp in zip(steps, step_inputs, strict=True)]
            return list(await asyncio.gather(*tasks))
        except Exception as e:  # pragma: no cover - handled in pipeline tests
            raise PipelineError("parallel", e) from e
