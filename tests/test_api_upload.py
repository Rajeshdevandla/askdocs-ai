"""API upload regressions using real PDF parsing without model downloads."""

from io import BytesIO
from pathlib import Path
import sys
from types import SimpleNamespace

from fastapi.testclient import TestClient
import numpy as np
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
import pytest


def pdf_bytes(*, encrypted=False, text=True):
    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=300)
    if text:
        font = DictionaryObject({
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        })
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})
        })
        content = DecodedStreamObject()
        content.set_data(b"BT /F1 12 Tf 20 200 Td (Annual revenue was 42 million.) Tj ET")
        page[NameObject("/Contents")] = content
    if encrypted:
        writer.encrypt("test-password")
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


@pytest.fixture
def upload_client(monkeypatch):
    # Only PDF parsing is under test; skip importing optional embedding libraries.
    for name, module in [
        ("faiss", SimpleNamespace()),
        ("sentence_transformers", SimpleNamespace(SentenceTransformer=object)),
    ]:
        if name not in sys.modules:
            monkeypatch.setitem(sys.modules, name, module)
    from api import main
    from core.rag_pipeline import RAGPipeline

    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.session_id = "upload-test-session"
    pipeline.document_names = []
    pipeline.document_loaded = False
    pipeline.embedder = SimpleNamespace(
        embed_chunks=lambda chunks: np.ones((len(chunks), 2), dtype=np.float32)
    )
    pipeline.vector_store = SimpleNamespace(add_chunks=lambda *args: None)
    paths = []
    original_load_pdf = pipeline.load_pdf

    def load_pdf(path, source_name=None):
        paths.append(Path(path))
        return original_load_pdf(path, source_name=source_name)

    monkeypatch.setattr(pipeline, "load_pdf", load_pdf)
    monkeypatch.setattr(main, "RAGPipeline", lambda: pipeline)
    monkeypatch.setattr(main, "sessions", {})
    with TestClient(main.app) as client:
        yield client, main.sessions, paths, pipeline


@pytest.mark.parametrize(
    "contents",
    [b"", b"not a PDF", b"%PDF-1.7\ntruncated", pdf_bytes(encrypted=True)],
    ids=["empty", "non-pdf", "truncated", "password-protected"],
)
def test_unreadable_pdf_is_a_validation_error_and_leaves_no_session(upload_client, contents):
    client, sessions, paths, _ = upload_client
    response = client.post("/upload", files={"file": ("report.pdf", contents, "application/pdf")})

    assert response.status_code == 422
    assert "readable PDF" in response.json()["detail"]
    assert sessions == {}
    assert paths and all(not path.exists() for path in paths)


def test_textless_pdf_remains_a_validation_error(upload_client):
    client, sessions, paths, _ = upload_client
    response = client.post("/upload", files={"file": ("scan.pdf", pdf_bytes(text=False), "application/pdf")})

    assert response.status_code == 422
    assert "No text found" in response.json()["detail"]
    assert sessions == {}
    assert all(not path.exists() for path in paths)


def test_valid_pdf_still_creates_a_session_and_removes_temp_file(upload_client):
    client, sessions, paths, pipeline = upload_client
    response = client.post("/upload", files={"file": ("report.pdf", pdf_bytes(), "application/pdf")})

    assert response.status_code == 200
    assert response.json() == {
        "session_id": pipeline.session_id,
        "document_name": "report.pdf",
        "page_count": 1,
        "chunk_count": 1,
    }
    assert sessions == {pipeline.session_id: pipeline}
    assert pipeline.document_loaded
    assert all(not path.exists() for path in paths)


def test_processing_failure_remains_a_server_error(upload_client, monkeypatch):
    client, sessions, paths, pipeline = upload_client

    def fail_embedding(chunks):
        raise RuntimeError("embedding backend unavailable")

    monkeypatch.setattr(pipeline.embedder, "embed_chunks", fail_embedding)
    response = client.post("/upload", files={"file": ("report.pdf", pdf_bytes(), "application/pdf")})

    assert response.status_code == 500
    assert response.json()["detail"] == "Failed to process PDF"
    assert sessions == {}
    assert paths and all(not path.exists() for path in paths)
