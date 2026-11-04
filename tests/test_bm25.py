import pytest

from app.bm25 import BM25Index
from app.rag import reciprocal_rank_fusion
from app.schemas import AskRequest, Chunk


@pytest.fixture
def index() -> BM25Index:
    idx = BM25Index()
    idx.build(
        [
            ("pto", "Employees receive 20 days of paid time off per year."),
            ("mfa", "Hardware security keys such as YubiKeys are required for production access."),
            ("expenses", "Submit expenses in the Expensify app within 30 days."),
            ("long", "Paid time off. " * 40),  # long chunk that repeats a phrase many times
        ]
    )
    return idx


def test_exact_term_ranks_first(index):
    results = index.search("Who needs YubiKeys?")
    assert results[0][0] == "mfa"


def test_rare_terms_outweigh_common_ones(index):
    # "expensify" appears in one chunk, "days" in several: the rare term should decide the ranking.
    assert index.search("expensify days")[0][0] == "expenses"


def test_length_normalization_limits_repetition(index):
    # Repeating "paid time off" 40 times must not beat a short chunk that also has "20 days".
    results = dict(index.search("20 days paid time off"))
    assert results["pto"] > results["long"]


def test_no_match_returns_empty(index):
    assert index.search("football world cup") == []
    assert index.search("the and of") == []  # only stopwords


def test_allowed_filter(index):
    assert [cid for cid, _ in index.search("days", allowed={"expenses"})] == ["expenses"]


def test_min_coverage_drops_single_word_matches(index):
    # Only 1 of 3 query terms ("days") appears in the PTO chunk.
    assert index.search("days submarine volcano", min_coverage=0.0)
    assert index.search("days submarine volcano", min_coverage=0.5) == []


def test_empty_index():
    assert BM25Index().search("anything") == []


def _chunk(cid: str) -> Chunk:
    return Chunk(id=cid, doc_id="d", filename="f.md", index=0, text=cid)


def test_rrf_rewards_agreement_between_lists():
    a, b, c = _chunk("a"), _chunk("b"), _chunk("c")
    vector = [(a, 0.9), (b, 0.8), (c, 0.1)]
    keyword = [(b, 12.0), (c, 3.0)]
    fused = reciprocal_rank_fusion([vector, keyword], top_k=3)
    # b and c appear in both lists, so they outrank a, which only one list found.
    assert [ch.id for ch, _ in fused] == ["b", "c", "a"]


def test_rrf_scores_are_normalized():
    a = _chunk("a")
    fused = reciprocal_rank_fusion([[(a, 0.5)], [(a, 7.0)]], top_k=1)
    assert fused[0][1] == pytest.approx(1.0)  # first in every list


@pytest.mark.parametrize("mode", ["vector", "bm25", "hybrid"])
def test_every_mode_finds_the_right_document(loaded_service, mode):
    hits = loaded_service.retrieve("How many PTO days do employees get?", top_k=3, mode=mode)
    assert hits[0][0].filename == "employee_handbook.md"


def test_hybrid_recovers_what_vector_search_misses(loaded_service):
    # A short query with a rare product name scores below MIN_SCORE in vector search,
    # but BM25 matches the exact word, so hybrid still finds it.
    assert loaded_service.retrieve("Who needs YubiKeys?", top_k=3, mode="vector") == []
    hits = loaded_service.retrieve("Who needs YubiKeys?", top_k=3, mode="hybrid")
    assert hits[0][0].filename == "it_security_policy.md"


def test_hybrid_finds_exact_terms_and_paraphrases(loaded_service):
    exact = loaded_service.retrieve("Which app do I submit Expensify receipts in?", top_k=4, mode="hybrid")
    paraphrase = loaded_service.retrieve("Is SMS allowed as a second factor for login?", top_k=4, mode="hybrid")
    assert exact[0][0].filename == "employee_handbook.md"
    assert paraphrase[0][0].filename == "it_security_policy.md"


def test_keyword_index_follows_uploads_and_deletes(service):
    doc, _ = service.ingest("notes.txt", b"The quarterly zeppelin inspection happens in hangar 7.")
    assert service.retrieve("zeppelin inspection", mode="bm25")
    service.store.delete(doc.doc_id)
    assert service.retrieve("zeppelin inspection", mode="bm25") == []


def test_hybrid_respects_doc_filter(loaded_service):
    faq = next(d for d in loaded_service.store.list_documents() if d.filename == "product_faq.md")
    hits = loaded_service.retrieve("security keys password", top_k=4, doc_ids=[faq.doc_id], mode="hybrid")
    assert {c.filename for c, _ in hits} <= {"product_faq.md"}


def test_request_can_override_mode(loaded_service):
    res = loaded_service.ask(AskRequest(question="Who needs YubiKeys?", retrieval_mode="bm25"))
    assert res.grounded
    assert res.citations[0].filename == "it_security_policy.md"
