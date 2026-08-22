import logging
import uuid
from pathlib import Path
from typing import Optional
from pypdf import PdfReader

from config import config
from core.embedder import Embedder
from core.llm_client import build_llm_client
from core.prompting import build_rag_prompt
from core.text_utils import split_into_chunks
from core.vector_store import VectorStore

logger = logging.getLogger(__name__)



class RAGPipeline:
    """
    The main orchestrator for the RAG (Retrieval-Augmented Generation) flow.

    Each user session gets its own pipeline instance so documents
    don't leak between different users.

    What it does:
    1. Parses PDFs and stores them in the vector store
    2. Takes user questions, finds relevant chunks, sends them to Bedrock
    3. Keeps a short conversation history so follow-up questions work
    """

    def __init__(self):
        self.session_id = str(uuid.uuid4())
        self.embedder = Embedder(model_name=config.embedding_model_name)
        self.vector_store = VectorStore(dimension=self.embedder.dimension)
        self.conversation_history: list[dict] = []
        self.document_loaded = False
        self.document_name: Optional[str] = None

        self.llm = build_llm_client(config)

        logger.info(f"RAGPipeline initialized, session={self.session_id}")

    def load_pdf(self, pdf_path: str) -> dict:
        """
        Parse a PDF file and load its content into the vector store.

        Goes page by page, splits each page into chunks,
        embeds them, and stores them.
        """
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        logger.info(f"Loading PDF: {path.name}")
        reader = PdfReader(str(path))

        all_chunks = []
        all_metadata = []

        for page_num, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text()

            if not page_text or not page_text.strip():
                continue

            chunks = split_into_chunks(
                text=page_text,
                chunk_size=config.chunk_size,
                overlap=config.chunk_overlap,
            )

            for chunk in chunks:
                all_chunks.append(chunk)
                all_metadata.append({"page": page_num, "source": path.name})

        if not all_chunks:
            raise ValueError(
                "No text found in this PDF. It might be a scanned image."
            )

        logger.info(f"Embedding {len(all_chunks)} chunks...")
        vectors = self.embedder.embed_chunks(all_chunks)
        self.vector_store.add_chunks(all_chunks, vectors, all_metadata)

        self.document_loaded = True
        self.document_name = path.name

        return {
            "document_name": path.name,
            "page_count": len(reader.pages),
            "chunk_count": len(all_chunks),
        }

    def ask(self, question: str) -> dict:
        """
        Answer a question using the loaded document.

        Flow:
        1. Embed the question
        2. Search vector store for relevant chunks
        3. Build a prompt with the retrieved context
        4. Call Bedrock Claude to generate an answer
        5. Save to history and return
        """
        if not self.document_loaded:
            raise RuntimeError("No document loaded. Upload a PDF first.")

        if not question.strip():
            raise ValueError("Question cannot be empty.")

        # embed and search
        query_vector = self.embedder.embed_query(question)
        results = self.vector_store.search(query_vector, top_k=config.top_k_results)

        if not results:
            return {
                "answer": "I couldn't find relevant content in the document to answer that.",
                "citations": [],
                "session_id": self.session_id,
            }

        # build context from retrieved chunks
        context = "\n\n".join(
            f"[Page {r['metadata'].get('page', '?')}]: {r['text']}" for r in results
        )

        prompt = build_rag_prompt(context, question, self.conversation_history)
        answer = self.llm.generate(prompt)

        self.conversation_history.append({"question": question, "answer": answer})

        return {
            "answer": answer,
            "citations": [
                {"page": r["metadata"].get("page"), "score": round(r["score"], 3)}
                for r in results
            ],
            "session_id": self.session_id,
        }

