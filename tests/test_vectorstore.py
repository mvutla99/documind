import pytest

from app.schemas import Chunk, DocumentInfo
from app.vectorstore import VectorStore


def _doc(doc_id, n):
    info = DocumentInfo(doc_id=doc_id, filename=f"{doc_id}.txt", content_hash=f"h-{doc_id}", num_chunks=n, num_pages=1)
    chunks = [
        Chunk(id=f"{doc_id}-{i}", doc_id=doc_id, filename=info.filename, index=i, text=f"{doc_id} {i}")
        for i in range(n)
    ]
    return info, chunks


def test_search_returns_most_similar_first(tmp_path):
    store = VectorStore(tmp_path)
    info, chunks = _doc("a", 3)
    store.add(info, chunks, [[1, 0, 0], [0, 1, 0], [0.9, 0.1, 0]])
    hits = store.search([1, 0, 0], k=2)
    assert [c.id for c, _ in hits] == ["a-0", "a-2"]
    assert hits[0][1] == pytest.approx(1.0)
    assert hits[0][1] >= hits[1][1]


def test_filter_by_document(tmp_path):
    store = VectorStore(tmp_path)
    a, ac = _doc("a", 1)
    b, bc = _doc("b", 1)
    store.add(a, ac, [[1, 0]])
    store.add(b, bc, [[0.8, 0.2]])
    hits = store.search([1, 0], k=5, doc_ids=["b"])
    assert [c.doc_id for c, _ in hits] == ["b"]


def test_delete_removes_vectors(tmp_path):
    store = VectorStore(tmp_path)
    a, ac = _doc("a", 2)
    b, bc = _doc("b", 1)
    store.add(a, ac, [[1, 0], [0, 1]])
    store.add(b, bc, [[1, 1]])
    assert store.delete("a") is True
    assert store.delete("a") is False
    assert len(store) == 1
    assert [c.doc_id for c, _ in store.search([1, 0], k=5)] == ["b"]


def test_persists_and_reloads(tmp_path):
    store = VectorStore(tmp_path)
    a, ac = _doc("a", 2)
    store.add(a, ac, [[1, 0], [0, 1]])
    reloaded = VectorStore(tmp_path)
    assert len(reloaded) == 2
    assert reloaded.get_document("a").filename == "a.txt"
    assert reloaded.search([0, 1], k=1)[0][0].id == "a-1"


def test_rejects_dimension_mismatch(tmp_path):
    store = VectorStore(tmp_path)
    a, ac = _doc("a", 1)
    b, bc = _doc("b", 1)
    store.add(a, ac, [[1, 0]])
    with pytest.raises(ValueError, match="dimension"):
        store.add(b, bc, [[1, 0, 0]])


def test_empty_store_search():
    assert VectorStore().search([1, 0], k=3) == []
