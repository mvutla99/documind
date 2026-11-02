import pytest

from app.ingestion.chunker import chunk_text

TEXT = "\n\n".join(
    f"Paragraph {p}. " + " ".join(f"Sentence {p}.{s} has some words in it." for s in range(8)) for p in range(10)
)


def test_chunks_respect_max_size():
    chunks = chunk_text(TEXT, chunk_size=300, overlap=50)
    assert len(chunks) > 5
    assert all(len(c) <= 300 for c in chunks)


def test_no_content_is_lost():
    chunks = chunk_text(TEXT, chunk_size=300, overlap=50)
    joined = " ".join(chunks)
    for p in range(10):
        for s in range(8):
            assert f"Sentence {p}.{s} " in joined


def test_consecutive_chunks_overlap():
    chunks = chunk_text(TEXT, chunk_size=300, overlap=80)
    a, b = chunks[0], chunks[1]
    assert a.split()[-1] in b.split()[:20]


def test_short_text_is_single_chunk():
    assert chunk_text("Just one line.", chunk_size=100, overlap=10) == ["Just one line."]


def test_empty_text_gives_no_chunks():
    assert chunk_text("   \n\n  ") == []


def test_very_long_sentence_is_hard_split():
    chunks = chunk_text("x" * 2500, chunk_size=1000, overlap=0)
    assert [len(c) for c in chunks] == [1000, 1000, 500]


@pytest.mark.parametrize("size,overlap", [(0, 0), (100, 100), (100, -1)])
def test_invalid_arguments(size, overlap):
    with pytest.raises(ValueError):
        chunk_text("hello", chunk_size=size, overlap=overlap)
