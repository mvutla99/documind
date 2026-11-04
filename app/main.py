"""FastAPI application.

Run:  uvicorn app.main:create_app --factory --reload
Docs: http://localhost:8000/docs
"""

from __future__ import annotations

import logging
import secrets
import time
from collections.abc import Iterator
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Request, Response, Security, UploadFile, status
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.security import APIKeyHeader
from starlette.concurrency import run_in_threadpool

from app import __version__
from app.config import Settings, get_settings
from app.factory import build_service
from app.ingestion import UnsupportedFileTypeError
from app.rag import EmptyDocumentError, RAGService
from app.schemas import AskRequest, AskResponse, DocumentInfo

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("documind")

STATIC_DIR = Path(__file__).parent / "static"
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def create_app(settings: Settings | None = None, service: RAGService | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(
        title="DocuMind",
        version=__version__,
        description="Upload documents and ask questions. Answers cite their sources (RAG).",
    )
    app.state.settings = settings
    app.state.service = service or build_service(settings)

    # ---------- dependencies ----------
    def get_service(request: Request) -> RAGService:
        return request.app.state.service

    def require_api_key(key: str | None = Security(_api_key_header)) -> None:
        expected = settings.api_key
        if expected and not (key and secrets.compare_digest(key, expected)):  # constant-time compare
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid X-API-Key header")

    protected = [Depends(require_api_key)]

    # ---------- middleware ----------
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        logger.info(
            "%s %s -> %d (%.0f ms)",
            request.method,
            request.url.path,
            response.status_code,
            (time.perf_counter() - start) * 1000,
        )
        return response

    # ---------- routes ----------
    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/health", tags=["meta"])
    def health(svc: RAGService = Depends(get_service)) -> dict:
        return {
            "status": "ok",
            "version": __version__,
            "embedder": svc.embedder.name,
            "llm": svc.llm.name,
            "retrieval": svc.retrieval_mode,
            "documents": len(svc.store.list_documents()),
            "chunks": len(svc.store),
        }

    @app.post(
        "/documents",
        response_model=DocumentInfo,
        status_code=status.HTTP_201_CREATED,
        dependencies=protected,
        tags=["documents"],
    )
    async def upload_document(
        response: Response, file: UploadFile = File(...), svc: RAGService = Depends(get_service)
    ):
        """Index a PDF, TXT or MD file. Re-uploading identical content returns the existing document (200)."""
        max_bytes = settings.max_upload_mb * 1024 * 1024
        data = await file.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, f"File exceeds {settings.max_upload_mb} MB")
        try:
            # Parsing + embedding is blocking work (CPU and network): run it off the event loop.
            doc, created = await run_in_threadpool(svc.ingest, file.filename or "upload.txt", data)
        except UnsupportedFileTypeError as e:
            raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, str(e)) from e
        except EmptyDocumentError as e:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(e)) from e
        if not created:
            response.status_code = status.HTTP_200_OK
        return doc

    @app.get("/documents", response_model=list[DocumentInfo], dependencies=protected, tags=["documents"])
    def list_documents(svc: RAGService = Depends(get_service)):
        return svc.store.list_documents()

    @app.delete(
        "/documents/{doc_id}", status_code=status.HTTP_204_NO_CONTENT, dependencies=protected, tags=["documents"]
    )
    def delete_document(doc_id: str, svc: RAGService = Depends(get_service)) -> None:
        if not svc.store.delete(doc_id):
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"Document '{doc_id}' not found")

    @app.post("/ask", response_model=AskResponse, dependencies=protected, tags=["qa"])
    def ask(req: AskRequest, svc: RAGService = Depends(get_service)):
        # A plain `def` endpoint: FastAPI runs it in a thread pool, so blocking LLM calls are fine.
        return svc.ask(req)

    @app.post("/ask/stream", dependencies=protected, tags=["qa"])
    def ask_stream(req: AskRequest, svc: RAGService = Depends(get_service)) -> StreamingResponse:
        """Newline-delimited JSON: one `citations` event, many `token` events, then `done`."""

        def events() -> Iterator[str]:
            try:
                for event in svc.ask_stream(req):
                    yield event.model_dump_json(exclude_none=True) + "\n"
            except Exception:
                logger.exception("Streaming failed")
                yield '{"type":"token","text":"\\n[Error: the model call failed. Please try again.]"}\n'
                yield '{"type":"done"}\n'

        return StreamingResponse(events(), media_type="application/x-ndjson")

    return app
