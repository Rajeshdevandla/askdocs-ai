from pathlib import Path

import pytest

from frontend.deployment import load_uploaded_pdf


class RecordingPipeline:
    def __init__(self):
        self.loaded_path = None
        self.contents = None

    def load_pdf(self, pdf_path):
        self.loaded_path = pdf_path
        self.contents = Path(pdf_path).read_bytes()
        return {"document_name": "sample.pdf", "page_count": 1, "chunk_count": 1}


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

        def load_pdf(self, pdf_path):
            self.loaded_path = pdf_path
            raise ValueError("invalid PDF")

    pipeline = FailingPipeline()

    with pytest.raises(ValueError, match="invalid PDF"):
        load_uploaded_pdf("broken.pdf", b"not-a-pdf", lambda: pipeline)

    assert not Path(pipeline.loaded_path).exists()
