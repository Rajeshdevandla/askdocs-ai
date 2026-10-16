"""Exercise both launch paths through real Streamlit reruns, without model calls."""

from io import BytesIO
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from pypdf import PdfReader, PdfWriter
import streamlit as st
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


class LocalPipeline:
    """Keep this UI regression independent of AWS and embedding downloads."""

    def __init__(self):
        self.documents = []
        self.questions = []

    def load_pdf(self, pdf_path, source_name=None):
        pages = len(PdfReader(pdf_path).pages)
        self.documents.append(source_name)
        return {"document_name": source_name, "page_count": pages, "chunk_count": 1}

    def ask(self, question):
        self.questions.append(question)
        return {
            "answer": f"Answer to: {question}",
            "citations": [{"source": self.documents[0], "page": 1, "score": 0.95}],
        }


@pytest.fixture
def uploads(monkeypatch):
    # AppTest lacks a file uploader driver on older supported Streamlit versions.
    # Replace only that widget and the model pipeline, running real session/UI code.
    selected = {}
    monkeypatch.setattr(
        st, "file_uploader", lambda *args, key, **kwargs: selected.get(key, [])
    )
    monkeypatch.setitem(
        sys.modules, "core.rag_pipeline", SimpleNamespace(RAGPipeline=LocalPipeline)
    )
    monkeypatch.delitem(sys.modules, "frontend.app", raising=False)
    yield selected
    sys.modules.pop("frontend.app", None)


def pdf_upload():
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    uploaded = BytesIO()
    writer.write(uploaded)
    uploaded.name = "annual-report.pdf"
    return uploaded


@pytest.mark.parametrize("entrypoint", ["streamlit_app.py", "frontend/app.py"])
def test_upload_questions_citations_and_reset_survive_reruns(uploads, entrypoint):
    app = AppTest.from_file(str(ROOT / entrypoint)).run()
    assert not app.exception
    assert len(app.title) == 1
    assert not app.chat_input

    uploads["pdf_upload_0"] = [pdf_upload()]
    app.run()
    assert not app.exception
    assert "1 document(s), 1 pages, and 1 chunks" in app.success[0].value
    pipeline = app.session_state.pipeline

    for count, question in enumerate(["What is revenue?", "What about costs?"], start=1):
        app.chat_input[0].set_value(question).run()
        assert not app.exception
        assert len(app.title) == 1
        assert len(app.chat_input) == 1
        assert len(app.chat_message) == count * 2
        assert len(app.expander) == count
        sources = [item.value for item in app.markdown
                   if "annual-report.pdf — Page 1" in item.value]
        assert len(sources) == count
        assert len(app.session_state.messages) == count * 2
        assert app.session_state.messages[-1]["content"] == f"Answer to: {question}"

    assert pipeline.questions == ["What is revenue?", "What about costs?"]
    assert pipeline.documents == ["annual-report.pdf"]  # No reindexing on questions.
    app.run()
    assert len(app.chat_message) == 4
    assert len(app.expander) == 2

    app.button[0].click().run()
    assert not app.exception
    assert len(app.title) == 1
    assert not app.chat_input
    assert not app.chat_message
    assert app.session_state.uploader_version == 1
    for key in ["pipeline", "document_hash", "document_names", "messages"]:
        assert key not in app.session_state

    uploads["pdf_upload_1"] = [pdf_upload()]
    app.run()
    assert not app.exception
    assert app.session_state.pipeline is not pipeline
    assert app.session_state.messages == []
    app.chat_input[0].set_value("New session question").run()
    assert not app.exception
    assert len(app.chat_message) == 2
    assert len(app.expander) == 1


def test_root_entrypoint_renders_for_a_second_session(uploads):
    uploads["pdf_upload_0"] = [pdf_upload()]
    first = AppTest.from_file(str(ROOT / "streamlit_app.py")).run()
    assert not first.exception
    assert len(first.chat_input) == 1

    uploads.clear()
    second = AppTest.from_file(str(ROOT / "streamlit_app.py")).run()
    assert not second.exception
    assert len(second.title) == 1
    assert not second.chat_input
    assert "pipeline" not in second.session_state
