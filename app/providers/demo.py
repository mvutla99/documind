"""Offline "demo mode" providers: no API key, no network, deterministic.

- HashingEmbedder: a bag-of-words vector using the hashing trick. Much weaker
  than a neural embedding model, but good enough to demo retrieval and to make
  tests fast and repeatable.
- ExtractiveLLM: instead of generating text, it picks the sentences from the
  retrieved sources that best match the question and cites them. It reads the
  same XML-tagged prompt a real LLM would get.
"""

from __future__ import annotations

import math
import re
import zlib
from collections.abc import Iterator

from app.text import tokenize


class HashingEmbedder:
    name = "demo-hashing"

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in tokenize(text):
            h = zlib.crc32(token.encode())  # stable across runs (unlike hash())
            vec[h % self.dim] += 1.0 if (h >> 16) & 1 else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


_SOURCE = re.compile(r'<source id="(\d+)"[^>]*>\s*(.*?)\s*</source>', re.S)
_QUESTION = re.compile(r"<question>\s*(.*?)\s*</question>", re.S)
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")

NO_ANSWER = "I couldn't find that in the uploaded documents."


class ExtractiveLLM:
    name = "demo-extractive"

    def __init__(self, max_sentences: int = 2) -> None:
        self.max_sentences = max_sentences

    def generate(self, system: str, prompt: str) -> str:
        q_match = _QUESTION.search(prompt)
        question = set(tokenize(q_match.group(1) if q_match else ""))
        scored: list[tuple[float, int, str]] = []
        for source_id, body in _SOURCE.findall(prompt):
            for sentence in _SENTENCE.split(body):
                sentence = sentence.strip().lstrip("#-* ").strip()
                words = set(tokenize(sentence))
                if len(words) < 3 or sentence[-1] not in ".!":  # skip headings, fragments and FAQ questions
                    continue
                overlap = len(question & words)
                if overlap:
                    rank_penalty = 1 + 0.15 * (int(source_id) - 1)  # trust higher-ranked sources more
                    scored.append((overlap / math.sqrt(len(words)) / rank_penalty, int(source_id), sentence))
        if not scored:
            return NO_ANSWER
        scored.sort(key=lambda s: s[0], reverse=True)
        best = scored[0][0]
        picked = [s for s in scored[: self.max_sentences] if s[0] >= 0.5 * best]
        return " ".join(f"{text} [{sid}]" for _, sid, text in picked)

    def stream(self, system: str, prompt: str) -> Iterator[str]:
        words = self.generate(system, prompt).split(" ")
        for i, word in enumerate(words):
            yield word if i == 0 else " " + word
