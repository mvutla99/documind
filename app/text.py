"""Shared text helpers used by keyword search (BM25) and the offline demo providers."""

from __future__ import annotations

import re

_WORD = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    "a an and are as at be by can do does for from has have how i if in is it its me my of on or "
    "our so that the their there this to us was we what when where which who why will with you your".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase words without stopwords, with crude stemming so "days"/"day" and "returned"/"return" match."""
    tokens = []
    for word in _WORD.findall(text.lower()):
        if word in STOPWORDS:
            continue
        if len(word) > 5 and word.endswith("ing"):
            word = word[:-3]
        elif len(word) > 5 and word.endswith("ed"):
            word = word[:-2]
        elif len(word) > 4 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        tokens.append(word)
    return tokens
