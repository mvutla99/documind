"""Google Gemini providers (chat + embeddings) using the `google-genai` SDK.

Get a free API key at https://aistudio.google.com/apikey
"""

from __future__ import annotations

from collections.abc import Iterator

from google import genai
from google.genai import errors, types
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

_RETRYABLE_CODES = {429, 500, 502, 503, 504}


def _is_retryable(exc: BaseException) -> bool:
    return isinstance(exc, errors.APIError) and getattr(exc, "code", None) in _RETRYABLE_CODES


# Rate limits and transient server errors are normal with LLM APIs:
# retry up to 4 times with exponential backoff (1s, 2s, 4s ... capped at 20s).
_retry = retry(
    retry=retry_if_exception(_is_retryable),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, max=20),
    reraise=True,
)


class GeminiEmbedder:
    def __init__(self, api_key: str, model: str = "gemini-embedding-001", batch_size: int = 100) -> None:
        self._client = genai.Client(api_key=api_key)
        self.model = model
        self.name = f"gemini:{model}"
        self.batch_size = batch_size

    @_retry
    def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        response = self._client.models.embed_content(
            model=self.model,
            contents=texts,
            config=types.EmbedContentConfig(task_type=task_type),
        )
        return [list(e.values) for e in response.embeddings]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            vectors.extend(self._embed(texts[i : i + self.batch_size], "RETRIEVAL_DOCUMENT"))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "RETRIEVAL_QUERY")[0]


class GeminiLLM:
    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash",
        temperature: float = 0.2,
        max_output_tokens: int = 1024,
    ) -> None:
        self._client = genai.Client(api_key=api_key)
        self.model = model
        self.name = f"gemini:{model}"
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

    def _config(self, system: str) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            system_instruction=system,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )

    @_retry
    def generate(self, system: str, prompt: str) -> str:
        response = self._client.models.generate_content(
            model=self.model, contents=prompt, config=self._config(system)
        )
        return (response.text or "").strip()

    def stream(self, system: str, prompt: str) -> Iterator[str]:
        # Not retried: retrying halfway through a stream would duplicate text.
        for chunk in self._client.models.generate_content_stream(
            model=self.model, contents=prompt, config=self._config(system)
        ):
            if chunk.text:
                yield chunk.text
