import logging
import faiss
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)


class VectorStore:
    """
    In-memory vector store backed by FAISS.

    Stores document chunks as vectors and lets you search for the
    most similar ones given a query vector.

    Important: everything lives in RAM. When the server restarts,
    all data is lost. For a real production app you'd persist the
    FAISS index to disk or use a managed service like Pinecone.
    For this portfolio project, in-memory is fine.
    """

    def __init__(self, dimension: int):
        self.dimension = dimension

        # IndexFlatIP = inner product (cosine similarity when vectors are normalized)
        self.index = faiss.IndexFlatIP(dimension)

        # keep the original text alongside the vectors so we can return it
        self.chunks: list[str] = []
        self.metadata: list[dict] = []

        logger.info(f"VectorStore ready, dimension={dimension}")

    def add_chunks(
        self,
        chunks: list[str],
        vectors: np.ndarray,
        metadata: Optional[list[dict]] = None,
    ):
        """Add text chunks and their embedding vectors to the store."""
        if len(chunks) != vectors.shape[0]:
            raise ValueError("chunks and vectors must have the same length")

        self.index.add(vectors)
        self.chunks.extend(chunks)

        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{} for _ in chunks])

        logger.info(f"Added {len(chunks)} chunks. Total in store: {len(self.chunks)}")

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[dict]:
        """
        Find the chunks most similar to the query vector.

        Returns a list of dicts with text, score, and metadata.
        Results with score <= 0 are filtered out - they are not
        semantically related and would confuse the LLM.
        """
        if self.index.ntotal == 0:
            logger.warning("search called on empty index")
            return []

        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_vector, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if score > 0 and idx != -1:
                results.append(
                    {
                        "text": self.chunks[idx],
                        "score": float(score),
                        "metadata": self.metadata[idx],
                    }
                )

        return results

    def clear(self):
        """Wipe everything - useful when loading a new document."""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunks = []
        self.metadata = []
        logger.info("VectorStore cleared")

    @property
    def total_chunks(self) -> int:
        return self.index.ntotal
