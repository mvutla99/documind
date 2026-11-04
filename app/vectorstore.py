"""A small, thread-safe vector store backed by NumPy and persisted to disk.

Search is exact cosine similarity: one matrix-vector product over all stored
chunks. That is fast up to roughly 100k chunks; past that you would move to an
approximate index (HNSW) or a database such as pgvector, Qdrant or Chroma.
The `VectorStore` methods here are the seam where that swap would happen.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import numpy as np

from app.schemas import Chunk, DocumentInfo


class VectorStore:
    def __init__(self, persist_dir: Path | str | None = None) -> None:
        self._dir = Path(persist_dir) if persist_dir else None
        self._lock = threading.RLock()
        self._vectors = np.zeros((0, 0), dtype=np.float32)
        self._chunks: list[Chunk] = []
        self._docs: dict[str, DocumentInfo] = {}
        # Bumped on every write so other indexes (e.g. BM25) know when to rebuild.
        self.version = 0
        if self._dir:
            self._dir.mkdir(parents=True, exist_ok=True)
            self._load()

    # ---------- writes ----------
    def add(self, doc: DocumentInfo, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        if not chunks:
            raise ValueError("cannot add a document with no chunks")
        new = _normalize(np.asarray(embeddings, dtype=np.float32))
        with self._lock:
            if self._vectors.size and new.shape[1] != self._vectors.shape[1]:
                raise ValueError(
                    f"embedding dimension {new.shape[1]} does not match store dimension "
                    f"{self._vectors.shape[1]}; did you change embedding models? Clear the data dir."
                )
            self._vectors = new if not self._vectors.size else np.vstack([self._vectors, new])
            self._chunks.extend(chunks)
            self._docs[doc.doc_id] = doc
            self.version += 1
            self._save()

    def delete(self, doc_id: str) -> bool:
        with self._lock:
            if doc_id not in self._docs:
                return False
            keep = [i for i, c in enumerate(self._chunks) if c.doc_id != doc_id]
            self._vectors = self._vectors[keep] if keep else np.zeros((0, 0), dtype=np.float32)
            self._chunks = [self._chunks[i] for i in keep]
            del self._docs[doc_id]
            self.version += 1
            self._save()
            return True

    # ---------- reads ----------
    def search(
        self, query: list[float], k: int = 4, doc_ids: list[str] | None = None
    ) -> list[tuple[Chunk, float]]:
        with self._lock:
            if not self._chunks:
                return []
            q = _normalize(np.asarray(query, dtype=np.float32).reshape(1, -1))[0]
            scores = self._vectors @ q  # cosine similarity, since rows are unit length
            if doc_ids is not None:
                allowed = set(doc_ids)
                mask = np.array([c.doc_id in allowed for c in self._chunks])
                scores = np.where(mask, scores, -np.inf)
            k = min(k, len(self._chunks))
            top = np.argpartition(-scores, k - 1)[:k]
            top = top[np.argsort(-scores[top])]
            return [(self._chunks[i], float(scores[i])) for i in top if np.isfinite(scores[i])]

    def snapshot(self) -> tuple[int, list[Chunk]]:
        """A consistent (version, chunks) pair, read under the lock."""
        with self._lock:
            return self.version, list(self._chunks)

    def get_document(self, doc_id: str) -> DocumentInfo | None:
        return self._docs.get(doc_id)

    def find_by_hash(self, content_hash: str) -> DocumentInfo | None:
        return next((d for d in self._docs.values() if d.content_hash == content_hash), None)

    def list_documents(self) -> list[DocumentInfo]:
        with self._lock:
            return sorted(self._docs.values(), key=lambda d: d.created_at)

    def __len__(self) -> int:
        return len(self._chunks)

    # ---------- persistence ----------
    def _save(self) -> None:
        if not self._dir:
            return
        _atomic_write(self._dir / "vectors.npy", lambda f: np.save(f, self._vectors), binary=True)
        _atomic_write(
            self._dir / "chunks.json",
            lambda f: json.dump([c.model_dump() for c in self._chunks], f),
        )
        _atomic_write(
            self._dir / "documents.json",
            lambda f: json.dump([d.model_dump(mode="json") for d in self._docs.values()], f, indent=2),
        )

    def _load(self) -> None:
        vec_path, chunk_path, doc_path = (self._dir / n for n in ("vectors.npy", "chunks.json", "documents.json"))
        if not (vec_path.exists() and chunk_path.exists() and doc_path.exists()):
            return
        self._vectors = np.load(vec_path)
        self._chunks = [Chunk(**c) for c in json.loads(chunk_path.read_text())]
        self._docs = {d["doc_id"]: DocumentInfo(**d) for d in json.loads(doc_path.read_text())}


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


def _atomic_write(path: Path, write, binary: bool = False) -> None:
    """Write to a temp file then rename, so a crash never leaves a half-written file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb" if binary else "w") as f:
        write(f)
    os.replace(tmp, path)
