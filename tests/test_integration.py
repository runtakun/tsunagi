from __future__ import annotations

import pytest

from tests.conftest import add_one, double, to_string
from tsunagi import Pipeline


@pytest.mark.asyncio
async def test_integration_pipeline() -> None:
    pipe = Pipeline("integration")
    result = await pipe.run(add_one >> double >> to_string, input=2)
    assert result == "6"
