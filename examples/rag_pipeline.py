"""RAG pipeline using OpenAI + Qdrant + Anthropic.

Requires OPENAI_API_KEY, QDRANT_URL/QDRANT_API_KEY, and ANTHROPIC_API_KEY.
Install optional SDKs: `pip install openai qdrant-client anthropic`.
"""

from __future__ import annotations

import asyncio

from tsunagi import Pipeline, StdoutTracer, step


@step
async def embed(text: str) -> dict[str, object]:
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:  # pragma: no cover - example only
        raise RuntimeError("Install openai to run this example") from exc

    client = AsyncOpenAI()
    resp = await client.embeddings.create(model="text-embedding-3-small", input=text)
    return {"query": text, "vector": resp.data[0].embedding}


@step
async def search(payload: dict[str, object]) -> dict[str, object]:
    try:
        from qdrant_client import AsyncQdrantClient
    except ImportError as exc:  # pragma: no cover - example only
        raise RuntimeError("Install qdrant-client to run this example") from exc

    client = AsyncQdrantClient()
    hits = await client.search(
        collection_name="documents",
        query_vector=payload["vector"],
        limit=3,
    )
    contexts = [hit.payload.get("text", "") for hit in hits]
    return {"query": payload["query"], "contexts": contexts}


@step
async def generate(data: dict[str, object]) -> str:
    try:
        from anthropic import AsyncAnthropic
    except ImportError as exc:  # pragma: no cover - example only
        raise RuntimeError("Install anthropic to run this example") from exc

    client = AsyncAnthropic()
    prompt = "\n\n".join(data.get("contexts", []))
    message = await client.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=300,
        messages=[
            {"role": "user", "content": f"Context:\n{prompt}\n\nQ: {data['query']}"}
        ],
    )
    return "\n".join(block.text for block in message.content if block.type == "text")


async def main() -> None:
    pipe = Pipeline("rag").use(StdoutTracer(verbose=True))
    answer = await pipe.run(embed >> search >> generate, input="What is Tsunagi?")
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())
