# DocuMind: Interview Guide

How to present this project, walk through the code, and answer follow-up questions.

---

## 1. Your 60-second pitch

> "I built DocuMind, a RAG service that lets you upload documents and ask questions about them, with answers that cite the exact source and page. It's a FastAPI backend with Pydantic models, Google Gemini for embeddings and generation, and a NumPy vector store. I focused on the things that matter in production: answers are grounded, and if nothing relevant is found it says so instead of hallucinating. Responses stream, LLM calls retry with backoff, and I measure retrieval quality with an evaluation script that runs in CI. It also has an offline demo mode, so the whole test suite runs without an API key."

## 2. Resume bullets (pick 2–3)

- Built a Retrieval-Augmented Generation (RAG) API with **FastAPI, Pydantic and Google Gemini** that answers questions over PDF/Markdown documents with page-level citations
- Designed a **provider-agnostic pipeline** (embedder/LLM interfaces) with an offline mode, enabling a fast, deterministic **pytest** suite with no external API calls
- Implemented **streaming responses**, retries with exponential backoff, duplicate detection via content hashing, and optional API-key auth
- Created a **retrieval evaluation harness** (Hit@k, MRR) on a labeled question set, enforced in **GitHub Actions CI**; containerized with **Docker**

## 3. Walk through one request (they will ask this)

**Upload** (`POST /documents` → `RAGService.ingest`):
1. Read the file, reject it if over the size limit (413) or the wrong type (415).
2. SHA-256 hash the bytes. If the same content is already indexed, return the existing document (no duplicate embeddings, no extra API cost).
3. Extract text per page (`pypdf` for PDFs) so citations can include page numbers.
4. Chunk: split on paragraphs, then sentences, pack up to 1000 characters, carry 150 characters of overlap.
5. Embed all chunks in batches (`RETRIEVAL_DOCUMENT` task type) and store vectors + metadata.

**Question** (`POST /ask` → `RAGService.ask`):
1. Embed the question (`RETRIEVAL_QUERY` task type).
2. Cosine similarity against every chunk (one matrix-vector multiply), take top-k, drop anything below `MIN_SCORE`.
3. If nothing is left, return "not found" **without calling the LLM**.
4. Build the prompt: system rules + sources in `<source id="1">` tags + the question.
5. The LLM answers with `[n]` citations; the API returns the answer plus citation metadata.

## 4. Likely follow-up questions

**Why chunk with overlap?**
A fact can straddle a chunk boundary. Overlap means the sentence before the cut also appears at the start of the next chunk, so either chunk can match.

**How did you choose chunk size?**
1000 characters (~250 tokens) is a common starting point: small enough to be specific, large enough to keep context. Then I compared settings with the eval script (`--chunk-size 500` vs `1000`) instead of guessing.

**Why not use a vector database?**
For this scale, exact search over a NumPy matrix is simpler, has no extra service to run, and is fast up to roughly 100k chunks. The `VectorStore` class is an interface: for millions of chunks I'd move to pgvector (if we already run Postgres) or Qdrant, using approximate (HNSW) indexes.

**How do you prevent hallucinations?**
Four layers: (1) only relevant chunks reach the model; (2) the system prompt requires citations and an exact "not found" sentence; (3) if retrieval finds nothing, the LLM isn't called at all; (4) low temperature (0.2).

**What about prompt injection?**
Documents are untrusted. Their text goes inside `<source>` tags and the system prompt says never to follow instructions found there. The model also has no tools, so injected instructions can't trigger actions. Next steps would be output filtering and per-user document isolation.

**How would you scale it?**
Run multiple API workers behind a load balancer, move the vector store to pgvector, move ingestion to a background queue (Celery or a cloud queue) so uploads return immediately, and cache answers for repeated questions.

**Why `def` and not `async def` for `/ask`?**
The Gemini SDK call I use is blocking. In an `async def` endpoint it would freeze the event loop for everyone. A plain `def` endpoint runs in FastAPI's thread pool. For uploads I use `async def` (to `await file.read()`) and push the blocking work into `run_in_threadpool`.

**What does the eval measure, and what would you add?**
Hit@k and MRR measure retrieval: did we fetch the right document, and how high did it rank. I'd add answer-level metrics (faithfulness and relevance, for example with RAGAS or an LLM judge) and a larger, harder question set with near-miss distractors.

**Why the demo mode?**
Tests that call a paid API are slow, flaky and cost money. The interfaces let me inject fakes, so 30+ tests run in under a second. It also lets reviewers try the project without a key.

**What would you do differently next time?**
Add hybrid search (keyword BM25 + vectors) for exact terms like product codes, add a re-ranker, and support conversation history for follow-up questions.

## 5. Demo script (5 minutes)

1. Show the README and architecture diagram (30s).
2. Run `pytest -q` and `python -m evals.run_eval` to show quality is measured (1 min).
3. Open the UI in Gemini mode, upload `sample_docs/`, ask:
   - "How many PTO days can I carry over, and until when?" (a multi-fact answer with citation)
   - "Can I paste customer data into ChatGPT?" (reasoning over policy)
   - "Who is the CEO?" (the grounded refusal)
4. Open `/docs` and show the typed request/response schemas.
5. Show `rag.py`: the "skip LLM when nothing found" branch and the prompt builder.

## 6. Make it yours (do at least one before interviews)

Interviewers can tell when a project was copied. Extend it in a way you can explain:
- Add **hybrid search** with BM25 (the `rank-bm25` package) and report the eval improvement.
- Add **conversation memory**: accept `history` in `AskRequest` and rewrite follow-up questions.
- Add a **pgvector** `VectorStore` implementation with docker-compose.
- Swap in a **second provider** (OpenAI or Claude) behind the same interfaces.
- Index **your own** documents (for example, public docs from a product you've worked with) and write your own eval questions.
