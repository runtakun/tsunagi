"""The @step decorator — core building block of Tsunagi.

A step wraps an async function to make it composable in a Pipeline.
The function itself contains all business logic (including SDK calls).
The step adds: naming, retry, timeout, and composability (>> operator).

CRITICAL DESIGN DECISION:
  The @step decorator does NOT modify the function's behavior when called directly.
  `await my_step("input")` behaves exactly like calling the raw function.
  Retry, timeout, and tracing only activate when the step runs inside a Pipeline.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar, overload

from tsunagi.errors import StepError
from tsunagi.errors import TimeoutError as TsunagiTimeoutError
from tsunagi.retry import NO_RETRY, RetryConfig

T = TypeVar("T")


class Step:
    """A composable async function with metadata for pipeline execution.

    Attributes:
        fn: The original async function.
        name: Step name (defaults to function name).
        retry: Retry configuration.
        timeout_seconds: Optional timeout in seconds.
    """

    def __init__(
        self,
        fn: Callable[..., Awaitable[Any]],
        *,
        name: str | None = None,
        retry: RetryConfig = NO_RETRY,
        timeout_seconds: float | None = None,
    ) -> None:
        if not inspect.iscoroutinefunction(fn):
            raise TypeError(
                f"@step requires an async function, got {type(fn).__name__}. "
                f"Hint: add 'async' before 'def {fn.__name__}'."
            )
        self.fn = fn
        self.name = name or fn.__name__
        self.retry = retry
        self.timeout_seconds = timeout_seconds
        functools.update_wrapper(self, fn)

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Direct call — behaves exactly like the original function.

        No retry, no timeout, no tracing. Just the raw function.
        This ensures steps are testable in isolation without framework overhead.
        """
        return await self.fn(*args, **kwargs)

    async def execute(self, *args: Any, **kwargs: Any) -> Any:
        """Execute with retry and timeout. Called by Pipeline, not by users directly.

        Raises:
            StepError: If all retry attempts are exhausted.
            TsunagiTimeoutError: If the step exceeds its timeout.
        """
        last_error: Exception | None = None

        for attempt in range(self.retry.max_attempts):
            try:
                if self.timeout_seconds is not None:
                    result = await asyncio.wait_for(
                        self.fn(*args, **kwargs),
                        timeout=self.timeout_seconds,
                    )
                else:
                    result = await self.fn(*args, **kwargs)
                return result

            except TimeoutError:
                raise TsunagiTimeoutError(self.name, self.timeout_seconds) from None

            except Exception as e:
                last_error = e
                is_last_attempt = attempt >= self.retry.max_attempts - 1

                if is_last_attempt or not self.retry.should_retry(e):
                    raise StepError(self.name, e, attempt + 1) from e

                delay = self.retry.get_delay(attempt)
                await asyncio.sleep(delay)

        # Should not reach here, but just in case
        raise StepError(
            self.name, last_error or RuntimeError("Unknown error"), self.retry.max_attempts
        )

    def __rshift__(self, other: Step) -> StepSequence:
        """Compose steps: step_a >> step_b creates a sequence."""
        if isinstance(other, StepSequence):
            return StepSequence([self] + other.steps)
        if isinstance(other, Step):
            return StepSequence([self, other])
        return NotImplemented

    def bind(self, **kwargs: Any) -> BoundStep:
        """Partially apply keyword arguments to this step.

        Usage:
            generate_step = generate.bind(model="claude-sonnet-4-5-20250929")
            # When pipeline calls generate_step(context), it becomes:
            # generate(context, model="claude-sonnet-4-5-20250929")
        """
        return BoundStep(self, kwargs)

    def __repr__(self) -> str:
        return f"Step({self.name})"


class BoundStep(Step):
    """A step with pre-bound keyword arguments."""

    def __init__(self, original: Step, bound_kwargs: dict[str, Any]) -> None:
        # Don't call super().__init__ — we delegate to the original
        self._original = original
        self._bound_kwargs = bound_kwargs
        self.fn = original.fn
        self.name = original.name
        self.retry = original.retry
        self.timeout_seconds = original.timeout_seconds

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        merged = {**self._bound_kwargs, **kwargs}
        return await self._original.fn(*args, **merged)

    async def execute(self, *args: Any, **kwargs: Any) -> Any:
        merged = {**self._bound_kwargs, **kwargs}

        # Temporarily swap fn to include bound kwargs
        original_fn = self._original.fn

        async def bound_fn(*a: Any, **kw: Any) -> Any:
            return await original_fn(*a, **{**merged, **kw})

        self._original.fn = bound_fn  # type: ignore[assignment]
        try:
            return await self._original.execute(*args)
        finally:
            self._original.fn = original_fn  # type: ignore[assignment]


class StepSequence:
    """An ordered sequence of steps created by the >> operator.

    Used internally by Pipeline.run() to determine execution order.
    """

    def __init__(self, steps: list[Step]) -> None:
        self.steps = steps

    def __rshift__(self, other: Step | StepSequence) -> StepSequence:
        if isinstance(other, StepSequence):
            return StepSequence(self.steps + other.steps)
        if isinstance(other, Step):
            return StepSequence(self.steps + [other])
        return NotImplemented

    def __repr__(self) -> str:
        return " >> ".join(s.name for s in self.steps)


# --- Decorator ---


@overload
def step(fn: Callable[..., Awaitable[Any]]) -> Step: ...


@overload
def step(
    *,
    name: str | None = None,
    retry: RetryConfig = NO_RETRY,
    timeout_seconds: float | None = None,
) -> Callable[[Callable[..., Awaitable[Any]]], Step]: ...


def step(
    fn: Callable[..., Awaitable[Any]] | None = None,
    *,
    name: str | None = None,
    retry: RetryConfig = NO_RETRY,
    timeout_seconds: float | None = None,
) -> Step | Callable[[Callable[..., Awaitable[Any]]], Step]:
    """Decorator to create a pipeline step from an async function.

    Can be used with or without arguments:

        @step
        async def embed(text: str) -> list[float]: ...

        @step(retry=RETRY_3X, timeout_seconds=30)
        async def embed(text: str) -> list[float]: ...

    The decorated function remains directly callable:
        result = await embed("hello")  # No framework overhead

    Framework features (retry, timeout, tracing) activate only inside Pipeline.run().
    """
    if fn is not None:
        return Step(fn, name=name, retry=retry, timeout_seconds=timeout_seconds)

    def decorator(f: Callable[..., Awaitable[Any]]) -> Step:
        return Step(f, name=name, retry=retry, timeout_seconds=timeout_seconds)

    return decorator
