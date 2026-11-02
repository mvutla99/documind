"""Turn uploaded files into plain text, page by page."""

from __future__ import annotations

import io
from pathlib import Path

from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


class UnsupportedFileTypeError(ValueError):
    pass


def extract_pages(filename: str, data: bytes) -> list[tuple[int | None, str]]:
    """Return a list of (page_number, text).

    PDFs keep real page numbers (1-based) so answers can cite "page 3".
    Text and Markdown files are a single "page" with number None.
    """
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{ext or 'none'}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if ext == ".pdf":
        reader = PdfReader(io.BytesIO(data))
        return [(i, page.extract_text() or "") for i, page in enumerate(reader.pages, start=1)]

    return [(None, data.decode("utf-8", errors="replace"))]
