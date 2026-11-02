"""Pydantic models shared by the RAG core and the API layer."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A piece of a document that gets embedded and stored."""

    id: str
    doc_id: str
    filename: str
    page: int | None = None
    index: int
    text: str


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    content_hash: str
    num_chunks: int
    num_pages: int
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, examples=["How many PTO days do I get?"])
    top_k: int = Field(4, ge=1, le=20, description="How many chunks to retrieve")
    doc_ids: list[str] | None = Field(None, description="Only search these documents")


class Citation(BaseModel):
    source: int = Field(..., description="The [n] number used in the answer")
    doc_id: str
    filename: str
    page: int | None = None
    score: float
    snippet: str


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    grounded: bool = Field(..., description="False when no relevant context was found")


class StreamEvent(BaseModel):
    """One line of the NDJSON stream returned by POST /ask/stream."""

    type: Literal["citations", "token", "done"]
    text: str | None = None
    citations: list[Citation] | None = None
    grounded: bool | None = None
