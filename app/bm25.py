"""BM25 keyword search, implemented from scratch.

Vector search matches meaning ("vacation" ~ "PTO") but often misses exact terms:
product names, error codes, IDs and numbers. BM25 is the classic keyword ranking
function used by search engines (Lucene, Elasticsearch). Combining both
("hybrid search") gets the strengths of each.

For each query term t in a chunk d:

    idf(t)   = ln(1 + (N - n_t + 0.5) / (n_t + 0.5))      # rare terms count more
    score   += idf(t) * f * (k1 + 1) / (f + k1 * (1 - b + b * |d| / avgdl))

f is how often t appears in d. k1 limits how much repeating a word helps;
b controls how much long chunks are penalized.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable

from app.text import tokenize


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._ids: list[str] = []
        self._tfs: list[Counter[str]] = []
        self._lens: list[int] = []
        self._idf: dict[str, float] = {}
        self._postings: dict[str, list[int]] = {}  # term -> positions of chunks containing it
        self._avgdl = 0.0

    def build(self, docs: Iterable[tuple[str, str]]) -> None:
        """Index (id, text) pairs, replacing anything indexed before."""
        self._ids, self._tfs, self._lens, self._postings = [], [], [], {}
        df: Counter[str] = Counter()
        for pos, (doc_id, text) in enumerate(docs):
            tf = Counter(tokenize(text))
            self._ids.append(doc_id)
            self._tfs.append(tf)
            self._lens.append(sum(tf.values()))
            df.update(tf.keys())
            for term in tf:
                self._postings.setdefault(term, []).append(pos)
        n = len(self._ids)
        self._avgdl = (sum(self._lens) / n) if n else 0.0
        self._idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def __len__(self) -> int:
        return len(self._ids)

    def search(
        self, query: str, k: int = 10, allowed: set[str] | None = None, min_coverage: float = 0.0
    ) -> list[tuple[str, float]]:
        """Return up to k (id, score) pairs, best first. Only chunks with a positive score are returned.

        allowed:      restrict results to these ids.
        min_coverage: fraction of distinct query terms a chunk must contain (0.5 = at least half).
                      Stops a single common word from pulling in an unrelated chunk.
        """
        terms = [t for t in dict.fromkeys(tokenize(query)) if t in self._idf]
        n_query_terms = len(set(tokenize(query)))
        if not terms or not self._avgdl:
            return []

        scores: dict[int, float] = {}
        matched: Counter[int] = Counter()
        for term in terms:  # only visit chunks that contain the term (inverted index)
            idf = self._idf[term]
            for pos in self._postings[term]:
                f = self._tfs[pos][term]
                norm = self.k1 * (1 - self.b + self.b * self._lens[pos] / self._avgdl)
                scores[pos] = scores.get(pos, 0.0) + idf * f * (self.k1 + 1) / (f + norm)
                matched[pos] += 1

        results = [
            (self._ids[pos], score)
            for pos, score in scores.items()
            if score > 0
            and matched[pos] / n_query_terms >= min_coverage
            and (allowed is None or self._ids[pos] in allowed)
        ]
        results.sort(key=lambda r: r[1], reverse=True)
        return results[:k]
