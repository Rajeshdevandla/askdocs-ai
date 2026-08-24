from pathlib import Path

import pytest

from frontend.deployment import load_uploaded_pdf, load_uploaded_pdfs


class RecordingPipeline:
    def __init__(self):
        self.loaded_path = None
        self.contents = None

    def load_pdf(self, pdf_path, source_name=None):
        self.loaded_path = pdf_path
        self.contents = Path(pdf_path).read_bytes()
        return {"document_name": source_name, "page_count": 1, "chunk_count": 1}


def test_load_uploaded_pdf_uses_isolated_temp_file_and_cleans_it_up():
    pipeline, metadata = load_uploaded_pdf(
        "sample.pdf",
        b"%PDF-demo",
        RecordingPipeline,
    )

    assert pipeline.contents == b"%PDF-demo"
    assert metadata["document_name"] == "sample.pdf"
    assert not Path(pipeline.loaded_path).exists()


def test_load_uploaded_pdf_cleans_up_when_pipeline_fails():
    class FailingPipeline:
        loaded_path = None

        def load_pdf(self, pdf_path, source_name=None):
            self.loaded_path = pdf_path
            raise ValueError("invalid PDF")

    pipeline = FailingPipeline()

    with pytest.raises(ValueError, match="invalid PDF"):
        load_uploaded_pdf("broken.pdf", b"not-a-pdf", lambda: pipeline)

    assert not Path(pipeline.loaded_path).exists()


def test_load_uploaded_pdfs_reuses_pipeline_and_cleans_all_temp_files():
    class MultiRecordingPipeline:
        def __init__(self):
            self.loaded_paths = []
            self.contents = []

        def load_pdf(self, pdf_path, source_name=None):
            self.loaded_paths.append(pdf_path)
            self.contents.append(Path(pdf_path).read_bytes())
            return {"document_name": source_name, "page_count": 1, "chunk_count": 2}

    pipeline, documents = load_uploaded_pdfs(
        [("first.pdf", b"first"), ("second.pdf", b"second")],
        MultiRecordingPipeline,
    )

    assert pipeline.contents == [b"first", b"second"]
    assert [document["document_name"] for document in documents] == [
        "first.pdf",
        "second.pdf",
    ]
    assert all(not Path(path).exists() for path in pipeline.loaded_paths)


def test_load_uploaded_pdfs_rejects_empty_uploads():
    with pytest.raises(ValueError, match="At least one PDF"):
        load_uploaded_pdfs([], RecordingPipeline)
