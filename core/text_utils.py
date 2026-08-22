"""Text-processing helpers used by the RAG ingestion pipeline."""


def split_into_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks with validated boundaries.

    overlap must be smaller than chunk_size; otherwise the cursor cannot
    advance and chunking would loop forever.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(text), step):
        chunk = text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)

    return chunks
