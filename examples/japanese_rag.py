"""Japanese-focused RAG pipeline example.

Demonstrates domain-specific preprocessing: PDF extraction, normalization,
sentence chunking, and embedding. Requires optional deps when executed:
`pip install pdfplumber fugashi openai` and relevant API keys.
"""

from __future__ import annotations

import asyncio

from tsunagi import Pipeline, step


@step
async def extract_pdf(path: str) -> str:
    try:
        import pdfplumber
    except ImportError as exc:  # pragma: no cover - example only
        raise RuntimeError("Install pdfplumber to run this example") from exc

    text_parts: list[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text_parts.append(page.extract_text() or "")
    return "\n".join(text_parts)


@step
async def normalize_japanese(text: str) -> str:
    try:
        import unicodedata
    except ImportError as exc:  # pragma: no cover - example only
        raise RuntimeError("Python stdlib missing?") from exc

    normalized = unicodedata.normalize("NFKC", text)
    return normalized.replace("\u3000", " ").strip()


@step
async def sentence_chunks(text: str) -> list[str]:
    try:
        import re
    except ImportError as exc:  # pragma: no cover - example only
        raise RuntimeError("Python stdlib missing?") from exc

    sentences = re.split(r"(?<=[。！？])\s+", text)
    return [s for s in sentences if s]


@step
async def embed_chunks(chunks: list[str]) -> list[list[float]]:
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:  # pragma: no cover - example only
        raise RuntimeError("Install openai to run this example") from exc

    client = AsyncOpenAI()
    resp = await client.embeddings.create(
        model="text-embedding-3-small",
        input=chunks,
    )
    return [item.embedding for item in resp.data]


async def main() -> None:
    pipe = Pipeline("jp-rag")
    vectors = await pipe.run(
        extract_pdf >> normalize_japanese >> sentence_chunks >> embed_chunks,
        input="./docs/source.pdf",
    )
    print(f"Embedded {len(vectors)} chunks")


if __name__ == "__main__":
    asyncio.run(main())
