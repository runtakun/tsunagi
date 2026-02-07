"""Framework-specific exceptions. Minimal set — SDK exceptions pass through unmodified."""


class TsunagiError(Exception):
    """Base exception for all Tsunagi errors."""

    pass


class StepError(TsunagiError):
    """Raised when a step fails after all retries are exhausted.

    Attributes:
        step_name: Name of the failed step.
        original: The original exception from user code / SDK.
        attempts: Number of attempts made.
    """

    def __init__(self, step_name: str, original: Exception, attempts: int) -> None:
        self.step_name = step_name
        self.original = original
        self.attempts = attempts
        super().__init__(
            f"Step '{step_name}' failed after {attempts} attempt(s): {original}"
        )


class PipelineError(TsunagiError):
    """Raised when a pipeline execution fails.

    Attributes:
        failed_step: Name of the step that caused the failure.
        original: The underlying StepError or exception.
    """

    def __init__(self, failed_step: str, original: Exception) -> None:
        self.failed_step = failed_step
        self.original = original
        super().__init__(f"Pipeline failed at step '{failed_step}': {original}")


class TimeoutError(TsunagiError):
    """Raised when a step or pipeline exceeds its timeout."""

    def __init__(self, step_name: str, timeout_seconds: float) -> None:
        self.step_name = step_name
        self.timeout_seconds = timeout_seconds
        super().__init__(
            f"Step '{step_name}' timed out after {timeout_seconds}s"
        )
