"""API tests. Uses FastAPI's TestClient (needs `httpx`, installed via requirements-dev.txt)."""

import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("pydantic_settings")
from fastapi.testclient import TestClient  # noqa: E402

from app.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402
from tests.conftest import SAMPLE_DIR  # noqa: E402


@pytest.fixture
def client(service, tmp_path):
    settings = Settings(provider="demo", data_dir=tmp_path, _env_file=None)
    return TestClient(create_app(settings=settings, service=service))


@pytest.fixture
def secured_client(service, tmp_path):
    settings = Settings(provider="demo", data_dir=tmp_path, api_key="s3cret", _env_file=None)
    return TestClient(create_app(settings=settings, service=service))


def _upload(client, name="employee_handbook.md", **kw):
    data = (SAMPLE_DIR / name).read_bytes()
    return client.post("/documents", files={"file": (name, data, "text/markdown")}, **kw)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_ui_is_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "DocuMind" in r.text


def test_upload_list_ask_delete_flow(client):
    r = _upload(client)
    assert r.status_code == 201
    doc_id = r.json()["doc_id"]

    assert _upload(client).status_code == 200  # duplicate -> existing doc
    assert [d["doc_id"] for d in client.get("/documents").json()] == [doc_id]

    r = client.post("/ask", json={"question": "How many PTO days do I get?"})
    body = r.json()
    assert r.status_code == 200
    assert body["grounded"] is True
    assert body["citations"][0]["filename"] == "employee_handbook.md"

    assert client.delete(f"/documents/{doc_id}").status_code == 204
    assert client.delete(f"/documents/{doc_id}").status_code == 404


def test_stream_returns_ndjson(client):
    _upload(client)
    r = client.post("/ask/stream", json={"question": "What is the meal allowance when traveling?"})
    assert r.status_code == 200
    events = [json.loads(line) for line in r.text.splitlines() if line]
    assert events[0]["type"] == "citations"
    assert events[-1]["type"] == "done"
    assert "$75" in "".join(e.get("text", "") for e in events if e["type"] == "token")


def test_unsupported_file_type(client):
    r = client.post("/documents", files={"file": ("photo.png", b"\x89PNG", "image/png")})
    assert r.status_code == 415


def test_validation_error_on_empty_question(client):
    assert client.post("/ask", json={"question": ""}).status_code == 422


def test_api_key_required_when_configured(secured_client):
    assert secured_client.get("/documents").status_code == 401
    assert secured_client.get("/documents", headers={"X-API-Key": "wrong"}).status_code == 401
    assert secured_client.get("/documents", headers={"X-API-Key": "s3cret"}).status_code == 200
    assert secured_client.get("/health").status_code == 200  # health stays public
