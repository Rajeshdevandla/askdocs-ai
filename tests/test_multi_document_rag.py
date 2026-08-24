import sys
from types import SimpleNamespace

import numpy as np

# Keep this unit test lightweight; production dependencies are exercised by CI's
# dependency install and Docker build.
sys.modules.setdefault("faiss", SimpleNamespace())
sys.modules.setdefault(
    "sentence_transformers",
    SimpleNamespace(SentenceTransformer=object),
)

from core.rag_pipeline import RAGPipeline


class FakeEmbedder:
    def embed_query(self, question):
        return np.array([[1.0, 0.0]], dtype=np.float32)


class FakeVectorStore:
    def search(self, query_vector, top_k):
        return [
            {
                "text": "Revenue increased.",
                "score": 0.91,
                "metadata": {"source": "annual-report.pdf", "page": 4},
            },
            {
                "text": "Risk remained stable.",
                "score": 0.82,
                "metadata": {"source": "risk-report.pdf", "page": 7},
            },
        ]


class RecordingLLM:
    def __init__(self):
        self.prompt = None

    def generate(self, prompt):
        self.prompt = prompt
        return "Combined answer"


def test_ask_preserves_document_identity_in_context_and_citations():
    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.session_id = "test-session"
    pipeline.embedder = FakeEmbedder()
    pipeline.vector_store = FakeVectorStore()
    pipeline.conversation_history = []
    pipeline.document_loaded = True
    pipeline.document_names = ["annual-report.pdf", "risk-report.pdf"]
    pipeline.llm = RecordingLLM()

    result = pipeline.ask("Compare revenue and risk")

    assert "Source: annual-report.pdf, Page 4" in pipeline.llm.prompt
    assert "Source: risk-report.pdf, Page 7" in pipeline.llm.prompt
    assert result["citations"] == [
        {"source": "annual-report.pdf", "page": 4, "score": 0.91},
        {"source": "risk-report.pdf", "page": 7, "score": 0.82},
    ]
