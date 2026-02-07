"""Execution context shared across steps within a single pipeline run."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StepTiming:
    """Timing information for a single step execution."""

    step_name: str
    started_at: float
    ended_at: float | None = None
    duration_ms: float | None = None
    error: Exception | None = None

    def finish(self, error: Exception | None = None) -> None:
        self.ended_at = time.monotonic()
        self.duration_ms = (self.ended_at - self.started_at) * 1000
        self.error = error


@dataclass
class Context:
    """Execution context for a pipeline run.

    Attributes:
        run_id: Unique identifier for this pipeline execution.
        pipeline_name: Name of the pipeline being executed.
        metadata: User-defined metadata dict. Steps can read/write freely.
        timings: Ordered list of step timing records.
    """

    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    pipeline_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    timings: list[StepTiming] = field(default_factory=list)

    def start_step(self, step_name: str) -> StepTiming:
        """Record the start of a step. Returns the StepTiming for later completion."""
        timing = StepTiming(step_name=step_name, started_at=time.monotonic())
        self.timings.append(timing)
        return timing

    @property
    def total_duration_ms(self) -> float:
        """Total duration of all completed steps in milliseconds."""
        return sum(t.duration_ms for t in self.timings if t.duration_ms is not None)

    @property
    def failed_steps(self) -> list[StepTiming]:
        """Steps that ended with an error."""
        return [t for t in self.timings if t.error is not None]

    def summary(self) -> dict[str, Any]:
        """Return a summary dict suitable for logging."""
        return {
            "run_id": self.run_id,
            "pipeline": self.pipeline_name,
            "total_duration_ms": round(self.total_duration_ms, 2),
            "steps": [
                {
                    "name": t.step_name,
                    "duration_ms": round(t.duration_ms, 2) if t.duration_ms else None,
                    "error": str(t.error) if t.error else None,
                }
                for t in self.timings
            ],
        }
