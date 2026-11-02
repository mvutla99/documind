import io

import pytest

from app.ingestion import UnsupportedFileTypeError
from app.rag import NO_CONTEXT_ANSWER, EmptyDocumentError, RAGService
from app.schemas import AskRequest


def test_ingest_creates_chunks(service):
    doc, created = service.ingest("notes.txt", b"Our office is in Austin. " * 50)
    assert created
    assert doc.num_chunks >= 1
    assert service.store.get_document(doc.doc_id) is not None


def test_duplicate_upload_is_skipped(service):
    first, _ = service.ingest("a.txt", b"Same content here.")
    second, created = service.ingest("copy-of-a.txt", b"Same content here.")
    assert not created
    assert second.doc_id == first.doc_id
    assert len(service.store.list_documents()) == 1


def test_rejects_unsupported_type(service):
    with pytest.raises(UnsupportedFileTypeError):
        service.ingest("image.png", b"\x89PNG")


def test_rejects_empty_document(service):
    with pytest.raises(EmptyDocumentError):
        service.ingest("empty.txt", b"   ")


@pytest.mark.parametrize(
    "question,expected_file,expected_text",
    [
        ("How many PTO days do employees get per year?", "employee_handbook.md", "20 days"),
        ("How long must passwords be?", "it_security_policy.md", "14 characters"),
        ("What is the product warranty on the SunPeak 400?", "product_faq.md", "25-year"),
    ],
)
def test_answers_cite_the_right_document(loaded_service, question, expected_file, expected_text):
    res = loaded_service.ask(AskRequest(question=question))
    assert res.grounded
    assert res.citations[0].filename == expected_file
    assert expected_text in res.answer
    assert "[1]" in res.answer or "[2]" in res.answer


def test_unrelated_question_is_not_grounded(loaded_service):
    res = loaded_service.ask(AskRequest(question="Who won the 1998 football world cup?"))
    assert not res.grounded
    assert res.answer == NO_CONTEXT_ANSWER
    assert res.citations == []


def test_doc_filter_limits_sources(loaded_service):
    faq = next(d for d in loaded_service.store.list_documents() if d.filename == "product_faq.md")
    res = loaded_service.ask(AskRequest(question="Can I return panels for a refund?", doc_ids=[faq.doc_id]))
    assert {c.filename for c in res.citations} == {"product_faq.md"}


def test_stream_sends_citations_then_tokens_then_done(loaded_service):
    events = list(loaded_service.ask_stream(AskRequest(question="What is the daily meal allowance?")))
    assert events[0].type == "citations" and events[0].citations
    assert events[-1].type == "done"
    text = "".join(e.text for e in events if e.type == "token")
    assert "$75" in text


def test_prompt_wraps_sources_in_tags(loaded_service):
    hits = loaded_service.retrieve("PTO carry over", top_k=2)
    prompt = RAGService.build_prompt("PTO carry over", hits)
    assert prompt.count("<source id=") == 2
    assert "<question>\nPTO carry over\n</question>" in prompt


def test_pdf_ingestion_keeps_page_numbers(service):
    canvas = pytest.importorskip("reportlab.pdfgen.canvas")
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    c.drawString(72, 720, "Page one talks about onboarding laptops.")
    c.showPage()
    c.drawString(72, 720, "Page two says the parking garage closes at 9 PM.")
    c.save()
    doc, _ = service.ingest("guide.pdf", buf.getvalue())
    assert doc.num_pages == 2
    res = service.ask(AskRequest(question="When does the parking garage close?", top_k=1))
    assert res.citations[0].page == 2
