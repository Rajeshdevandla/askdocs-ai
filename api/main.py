import logging
import os
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core.rag_pipeline import RAGPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AskDocs AI",
    description="Upload a PDF and chat with it using Amazon Bedrock",
    version="1.0.0",
)

# allow the Streamlit frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # lock this down in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# store active sessions in memory
# for production you'd use Redis or a database
sessions: dict[str, RAGPipeline] = {}


class QuestionRequest(BaseModel):
    session_id: str
    question: str


@app.get("/health")
def health():
    return {"status": "ok", "service": "askdocs-ai"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF and process it for Q&A.

    Returns a session_id - include this in all /ask requests.
    """
    if not file.filename or not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # save to a temp file so pypdf can read it
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        pipeline = RAGPipeline()
        result = pipeline.load_pdf(tmp_path)
        sessions[pipeline.session_id] = pipeline

        return {
            "session_id": pipeline.session_id,
            "document_name": result["document_name"],
            "page_count": result["page_count"],
            "chunk_count": result["chunk_count"],
        }
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to process PDF")
    finally:
        os.unlink(tmp_path)


@app.post("/ask")
def ask_question(request: QuestionRequest):
    """Ask a question about the uploaded document."""
    pipeline = sessions.get(request.session_id)
    if not pipeline:
        raise HTTPException(
            status_code=404,
            detail="Session not found. Please upload a PDF first.",
        )

    try:
        result = pipeline.ask(request.question)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions/{session_id}")
def get_session_info(session_id: str):
    """Get info about a session - mostly for debugging."""
    pipeline = sessions.get(session_id)
    if not pipeline:
        raise HTTPException(status_code=404, detail="Session not found")

    return {
        "session_id": session_id,
        "document_name": pipeline.document_name,
        "chunk_count": pipeline.vector_store.total_chunks,
        "conversation_turns": len(pipeline.conversation_history),
    }
