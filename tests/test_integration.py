from __future__ import annotations

import pytest

from tsunagi import Pipeline
from tests.conftest import add_one, double, to_string


@pytest.mark.asyncio
async def test_integration_pipeline() -> None:
    pipe = Pipeline("integration")
    result = await pipe.run(add_one >> double >> to_string, input=2)
    assert result == "6"
