from pathlib import Path

import pytest

from app.providers.demo import ExtractiveLLM, HashingEmbedder
from app.rag import RAGService
from app.vectorstore import VectorStore

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_docs"


@pytest.fixture
def service(tmp_path) -> RAGService:
    return RAGService(
        embedder=HashingEmbedder(),
        llm=ExtractiveLLM(),
        store=VectorStore(tmp_path / "store"),
        chunk_size=600,
        chunk_overlap=100,
        min_score=0.1,
    )


@pytest.fixture
def loaded_service(service) -> RAGService:
    for path in sorted(SAMPLE_DIR.glob("*.md")):
        service.ingest(path.name, path.read_bytes())
    return service
