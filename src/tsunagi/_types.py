"""Internal type aliases used across the framework."""

from __future__ import annotations

from typing import Any, ParamSpec, TypeVar

T = TypeVar("T")
U = TypeVar("U")
P = ParamSpec("P")

# A step function signature: async (input) -> output
# We keep this loose intentionally — steps can have any signature.
StepResult = Any
