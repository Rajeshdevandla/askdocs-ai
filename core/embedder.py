import logging
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class Embedder:
    """
    Wraps the sentence-transformer model for converting text to vectors.

    The model is loaded once when the class is initialized, not per-request.
    Loading takes 2-3 seconds so doing it on every request would be too slow.
    This pattern (load once, reuse) is called the singleton pattern.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model ready. Vector dimension: {self.dimension}")

    def embed_chunks(self, texts: list[str]) -> np.ndarray:
        """
        Convert a list of text strings into embedding vectors.

        We normalize the vectors to unit length so that FAISS inner product
        search gives us cosine similarity scores between 0 and 1.
        """
        if not texts:
            raise ValueError("texts list is empty")

        vectors = self.model.encode(texts, show_progress_bar=False)
        vectors = vectors.astype(np.float32)

        # normalize so cosine similarity works correctly
        faiss.normalize_L2(vectors)

        return vectors

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single user question.

        Must apply the same normalization as embed_chunks - if we skip this,
        the similarity scores between query and chunks will be wrong.
        """
        if not query or not query.strip():
            raise ValueError("query cannot be empty")

        vector = self.model.encode([query])
        vector = vector.astype(np.float32)
        faiss.normalize_L2(vector)

        return vector
