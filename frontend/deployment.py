"""Deployment helpers shared by the Streamlit UI and unit tests."""

from collections.abc import Callable
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Protocol


class PdfPipeline(Protocol):
    def load_pdf(self, pdf_path: str) -> dict:
        """Load a PDF from a local path."""


def load_uploaded_pdf(
    file_name: str,
    contents: bytes,
    pipeline_factory: Callable[[], PdfPipeline],
) -> tuple[PdfPipeline, dict]:
    """Load uploaded bytes into a new isolated pipeline and remove the temporary PDF."""
    suffix = Path(file_name).suffix or ".pdf"
    temp_path: str | None = None

    try:
        with NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
            temp_file.write(contents)
            temp_path = temp_file.name

        pipeline = pipeline_factory()
        metadata = pipeline.load_pdf(temp_path)
        return pipeline, metadata
    finally:
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)
