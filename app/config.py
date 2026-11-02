"""Settings loaded from environment variables or a .env file (see .env.example)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # "demo" runs fully offline with no API key; "gemini" uses Google's models.
    provider: Literal["demo", "gemini"] = "demo"
    gemini_api_key: str | None = None
    gemini_chat_model: str = "gemini-2.5-flash"
    gemini_embedding_model: str = "gemini-embedding-001"
    temperature: float = Field(0.2, ge=0, le=2)

    data_dir: Path = Path("data")
    chunk_size: int = Field(1000, gt=100)
    chunk_overlap: int = Field(150, ge=0)
    # Chunks scoring below this cosine similarity are ignored. Tune it with evals/run_eval.py.
    # None = provider default (demo 0.1, gemini 0.0).
    min_score: float | None = None
    max_upload_mb: int = Field(20, gt=0)

    # If set, every API call must send the header "X-API-Key: <value>".
    api_key: str | None = None

    @model_validator(mode="after")
    def _check(self) -> Settings:
        if self.provider == "gemini" and not self.gemini_api_key:
            raise ValueError("PROVIDER=gemini requires GEMINI_API_KEY (get one at https://aistudio.google.com/apikey)")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return self

    @property
    def resolved_min_score(self) -> float:
        if self.min_score is not None:
            return self.min_score
        return 0.1 if self.provider == "demo" else 0.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
