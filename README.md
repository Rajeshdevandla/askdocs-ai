# AskDocs AI 📄

A RAG-based document assistant that lets you upload PDFs and chat with them using natural language. Built with Amazon Bedrock (Claude) for generation and FAISS for vector search.

---

## What this does

You upload a PDF. The app splits it into chunks, embeds each chunk using a local sentence-transformer model, and stores the vectors in a FAISS index. When you ask a question, the app embeds your question, finds the most similar chunks, and sends them to Amazon Bedrock (Claude) as context to generate a grounded answer with page citations.

This is called Retrieval-Augmented Generation (RAG). The main benefit over just passing the whole PDF to an LLM is that it works on documents of any length, and every answer is tied back to specific pages in the document.

## Architecture

```
User Question
      │
      ▼
  Embedder  ──── embed_query() ────▶  FAISS Index
  (local model)                           │
                                          │ top-k chunks
                                          ▼
                                   RAG Pipeline
                                          │
                                          │ prompt + context
                                          ▼
                                  Amazon Bedrock
                                  (Claude Haiku)
                                          │
                                          ▼
                                   Answer + Citations
```

**PDF ingestion flow:**
`PDF → pypdf → text chunks → embedder → FAISS index`

**Query flow:**
`question → embed → FAISS search → top chunks → Bedrock prompt → answer`

## Tech stack

| Component | Technology |
|-----------|-----------|
| LLM | Amazon Bedrock (Claude 3 Haiku) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2, local) |
| Vector store | FAISS (in-memory) |
| API | FastAPI + uvicorn |
| Frontend | Streamlit |
| PDF parsing | pypdf |
| Containerization | Docker |

## Project structure

```
askdocs-ai/
├── config.py              # all env vars loaded and validated here
├── core/
│   ├── embedder.py        # sentence-transformer wrapper
│   ├── vector_store.py    # FAISS wrapper
│   └── rag_pipeline.py    # PDF parsing + RAG orchestration
├── api/
│   └── main.py            # FastAPI routes
├── frontend/
│   └── app.py             # Streamlit UI
├── requirements.txt
├── .env.example
└── Dockerfile
```

## Getting started

**1. Clone the repo**

```bash
git clone https://github.com/Rajeshdevandla/askdocs-ai.git
cd askdocs-ai
```

**2. Set up environment variables**

```bash
cp .env.example .env
# fill in your AWS credentials in .env
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Start the API**

```bash
uvicorn api.main:app --reload
```

**5. Start the frontend** (in a separate terminal)

```bash
streamlit run frontend/app.py
```

Then open http://localhost:8501, upload a PDF, and start asking questions.

## Running with Docker

```bash
docker build -t askdocs-ai .
docker run -p 8000:8000 --env-file .env askdocs-ai
```

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /health | Health check |
| POST | /upload | Upload a PDF, get back a session_id |
| POST | /ask | Ask a question (requires session_id) |
| GET | /sessions/{id} | Get session info |

**Upload example:**
```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@my-document.pdf"
```

**Ask example:**
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"session_id": "your-session-id", "question": "What is the main topic?"}'
```

## Environment variables

| Variable | Description | Default |
|----------|-------------|---------|
| AWS_REGION | AWS region | us-east-1 |
| AWS_ACCESS_KEY_ID | AWS access key | required |
| AWS_SECRET_ACCESS_KEY | AWS secret key | required |
| BEDROCK_MODEL_ID | Bedrock model to use | claude-3-haiku |
| EMBEDDING_MODEL | Local embedding model | all-MiniLM-L6-v2 |
| CHUNK_SIZE | Characters per chunk | 500 |
| CHUNK_OVERLAP | Overlap between chunks | 50 |
| TOP_K_RESULTS | Chunks retrieved per query | 5 |

## Notes

- The embedding model runs locally — no additional API cost
- The FAISS index is in-memory, so sessions are lost on server restart
- For production, you'd want to persist the index to S3 or swap FAISS for Pinecone
- Each uploaded PDF creates its own session, so multiple users are isolated

## AWS setup required

You need an AWS account with:
1. IAM user with `bedrock:InvokeModel` permission
2. Amazon Bedrock model access enabled for Claude 3 Haiku in your region

---

Built as part of a GenAI portfolio for cloud/AI engineering roles.
