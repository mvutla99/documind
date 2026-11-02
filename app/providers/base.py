"""Provider interfaces.

The RAG pipeline depends on these small Protocols, not on a vendor SDK.
That makes it easy to swap Gemini for OpenAI/Claude/a local model, and lets
tests run with fast, offline fakes.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol


class Embedder(Protocol):
    name: str

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class LLM(Protocol):
    name: str

    def generate(self, system: str, prompt: str) -> str: ...

    def stream(self, system: str, prompt: str) -> Iterator[str]: ...
