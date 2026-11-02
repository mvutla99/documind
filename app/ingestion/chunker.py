"""Split text into overlapping chunks that respect paragraph and sentence boundaries.

Why not just cut every N characters? Cutting mid-sentence hurts retrieval:
a chunk that starts halfway through a sentence embeds poorly and reads badly
when shown to the LLM. So we split on the largest natural boundary that fits
(paragraph, then sentence, then hard cut as a last resort), pack those pieces
into chunks up to `chunk_size`, and carry a little text over between chunks
(`overlap`) so facts that straddle a boundary are not lost.
"""

from __future__ import annotations

import re

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")


def _split_units(text: str, max_len: int) -> list[str]:
    units: list[str] = []
    for para in _PARAGRAPH_BREAK.split(text):
        para = re.sub(r"[ \t]+", " ", para).strip()
        if not para:
            continue
        if len(para) <= max_len:
            units.append(para)
            continue
        for sentence in _SENTENCE_END.split(para):
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(sentence) <= max_len:
                units.append(sentence)
            else:  # a giant "sentence" (tables, logs): hard cut
                units.extend(sentence[i : i + max_len] for i in range(0, len(sentence), max_len))
    return units


def _tail(text: str, n: int) -> str:
    """Last ~n characters of text, starting on a word boundary."""
    if n <= 0 or not text:
        return ""
    tail = text[-n:]
    space = tail.find(" ")
    if 0 <= space < len(tail) - 1 and len(text) > n:
        tail = tail[space + 1 :]
    return tail.strip()


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    """Split `text` into chunks of at most `chunk_size` characters."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    chunks: list[str] = []
    current = ""
    for unit in _split_units(text, chunk_size):
        candidate = f"{current}\n{unit}" if current else unit
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        chunks.append(current)
        carry = _tail(current, overlap)
        current = f"{carry}\n{unit}" if carry and len(carry) + 1 + len(unit) <= chunk_size else unit
    if current:
        chunks.append(current)
    return chunks
