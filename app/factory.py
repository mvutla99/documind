"""Wire concrete providers together based on settings."""

from __future__ import annotations

from app.config import Settings
from app.rag import RAGService
from app.vectorstore import VectorStore


def build_service(settings: Settings) -> RAGService:
    if settings.provider == "gemini":
        from app.providers.gemini import GeminiEmbedder, GeminiLLM  # imported lazily: optional dependency

        embedder = GeminiEmbedder(settings.gemini_api_key, settings.gemini_embedding_model)
        llm = GeminiLLM(settings.gemini_api_key, settings.gemini_chat_model, settings.temperature)
    else:
        from app.providers.demo import ExtractiveLLM, HashingEmbedder

        embedder, llm = HashingEmbedder(), ExtractiveLLM()

    # Separate folder per embedding model: vectors from different models can't be mixed.
    store_dir = settings.data_dir / embedder.name.replace(":", "_").replace("/", "_")
    return RAGService(
        embedder=embedder,
        llm=llm,
        store=VectorStore(store_dir),
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        min_score=settings.resolved_min_score,
    )
