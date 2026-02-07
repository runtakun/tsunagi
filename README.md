# Tsunagi

SDK-first orchestration framework for AI pipelines and agents. Tsunagi keeps SDK calls
as plain user code while adding composition, retry, timeout, and tracing.

## Installation

```bash
pip install -e .
# or with dev tools
pip install -e .[dev]
```

## Quick Start

```python
from tsunagi import Pipeline, step


@step
async def add_one(x: int) -> int:
    return x + 1


@step
async def double(x: int) -> int:
    return x * 2


pipe = Pipeline("basic")
result = await pipe.run(add_one >> double, input=3)
print(result)  # 8
```

## Design Philosophy

- **SDK-first**: You call OpenAI/Anthropic/Qdrant directly. Tsunagi never wraps SDKs.
- **Opt-in extras**: Tracing, retry, and timeouts are explicit. No hidden behavior.
- **Zero runtime deps**: Core relies only on Python standard library.
- **Small pieces**: Decorators and adapters stay short and readable (<400 lines).

## API Highlights

- `@step`: Wrap async functions for pipeline use; direct calls stay untouched.
- `Pipeline`: Run steps sequentially or in parallel with tracing and context.
- `RetryConfig`: Per-step retry/backoff configuration.
- `Tracer`: Protocol for observability; includes `NullTracer` and `StdoutTracer`.
- `@tool` and `Agent`: Register async tools and drive LLM tool-use loops via adapters.

## Comparing with LangChain

- Tsunagi keeps SDK calls unwrapped; LangChain introduces wrappers and abstractions.
- No built-in chunking, parsing, or vector store layers — your code, your logic.
- Minimal surface area: a handful of decorators and classes instead of large class hierarchies.

## Examples

See `examples/` for pipelines, agents with tools, custom tracers, and RAG sketches. API-key
dependent examples are annotated accordingly.
