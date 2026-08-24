"""Deployment helpers shared by the Streamlit UI and unit tests."""

from collections.abc import Callable
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Protocol


class PdfPipeline(Protocol):
    def load_pdf(self, pdf_path: str, source_name: str | None = None) -> dict:
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
        metadata = pipeline.load_pdf(temp_path, source_name=file_name)
        return pipeline, metadata
    finally:
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)


def load_uploaded_pdfs(
    uploads: list[tuple[str, bytes]],
    pipeline_factory: Callable[[], PdfPipeline],
) -> tuple[PdfPipeline, list[dict]]:
    """Load multiple uploaded PDFs into one isolated RAG pipeline."""
    if not uploads:
        raise ValueError("At least one PDF is required")

    pipeline = pipeline_factory()
    metadata = []
    temp_paths: list[Path] = []

    try:
        for file_name, contents in uploads:
            suffix = Path(file_name).suffix or ".pdf"
            with NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
                temp_file.write(contents)
                temp_path = Path(temp_file.name)
            temp_paths.append(temp_path)
            result = pipeline.load_pdf(str(temp_path), source_name=file_name)
            metadata.append(result)
        return pipeline, metadata
    finally:
        for temp_path in temp_paths:
            temp_path.unlink(missing_ok=True)
