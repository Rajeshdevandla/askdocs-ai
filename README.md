# AskDocs AI

> **Upload any PDF and chat with it using natural language** — powered by Amazon Bedrock (Claude) and FAISS vector search.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Amazon Bedrock](https://img.shields.io/badge/powered%20by-Amazon%20Bedrock-orange.svg)](https://aws.amazon.com/bedrock/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED.svg)](https://www.docker.com/)

---

## Live Demo

> 🚀 **[Coming soon — deploying to Hugging Face Spaces]**
>
> To run locally, follow the [Quick Start](#quick-start) below.

---

## What Problem This Solves

Reading long PDFs to find one specific answer is slow and tedious. AskDocs AI lets you drop in any PDF — a research paper, legal contract, insurance policy, or manual — and ask it questions in plain English. Every answer is grounded in the actual document and cites the exact page it came from.

---

## Demo

```
User uploads: annual_report_2024.pdf

Q: "What was the total revenue in Q3?"
A: "Total revenue in Q3 2024 was $4.2 billion, a 12% increase year-over-year.
   [Source: Page 14, Financial Highlights section]"

Q: "What are the key risk factors mentioned?"
A: "The report identifies three primary risk factors: macroeconomic uncertainty,
   supply chain disruptions, and regulatory changes in the EU market.
   [Source: Page 31, Risk Factors section]"
```

---

## Architecture

```
PDF INGESTION FLOW
──────────────────
PDF File
   │
   ▼
pypdf (text extraction)
   │
   ▼
Text Chunker (500 chars, 50 overlap)
   │
   ▼
sentence-transformers (all-MiniLM-L6-v2)   ← runs locally, no API cost
   │  embed each chunk
   ▼
FAISS Index (in-memory vector store)
   │
   ▼
Session stored → ready for queries

QUERY FLOW
──────────
User Question
   │
   ▼
sentence-transformers (embed question)
   │
   ▼
FAISS similarity search → top-K chunks retrieved
   │
   ▼
Prompt assembled: [chunks + question]
   │
   ▼
Amazon Bedrock (Claude 3 Haiku)
   │
   ▼
Answer + Page Citations → User
```

**Key design decisions:**
- Embeddings run **locally** (no API cost, no latency on uploads)
- Each PDF gets its own **session** — multiple users are fully isolated
- FAISS is **in-memory** — fast for demos, swap to Pinecone/pgvector for production

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM | Amazon Bedrock (Claude 3 Haiku) |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` (local) |
| Vector Store | FAISS (in-memory) |
| API | FastAPI + uvicorn |
| Frontend | Streamlit |
| PDF Parsing | pypdf |
| Containerization | Docker |

---

## Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/Rajeshdevandla/askdocs-ai.git
cd askdocs-ai
```

### 2. Set up environment variables

```bash
cp .env.example .env
# Edit .env and fill in your AWS credentials
```

Required variables:

| Variable | Description |
|---|---|
| `AWS_REGION` | AWS region (e.g. `us-east-1`) |
| `AWS_ACCESS_KEY_ID` | IAM user access key |
| `AWS_SECRET_ACCESS_KEY` | IAM user secret key |

> **AWS setup:** You need an IAM user with `bedrock:InvokeModel` permission and Claude 3 Haiku enabled in Amazon Bedrock for your region.

### 3. Install and run

```bash
pip install -r requirements.txt

# Terminal 1 — start the API
uvicorn api.main:app --reload

# Terminal 2 — start the frontend
streamlit run frontend/app.py
```

Open **http://localhost:8501**, upload a PDF, and ask questions.

### Run with Docker

```bash
docker build -t askdocs-ai .
docker run -p 8000:8000 --env-file .env askdocs-ai
```

---

## API Reference

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/upload` | Upload a PDF → returns `session_id` |
| `POST` | `/ask` | Ask a question (requires `session_id`) |
| `GET` | `/sessions/{id}` | Get session metadata |

**Upload a PDF:**
```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@my-document.pdf"
# returns: { "session_id": "abc123" }
```

**Ask a question:**
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"session_id": "abc123", "question": "What is the main topic?"}'
```

---

## Project Structure

```
askdocs-ai/
├── config.py                  # env vars, loaded and validated at startup
├── core/
│   ├── embedder.py            # sentence-transformer wrapper
│   ├── vector_store.py        # FAISS wrapper (add/search)
│   └── rag_pipeline.py        # PDF parsing + full RAG orchestration
├── api/
│   └── main.py                # FastAPI routes
├── frontend/
│   └── app.py                 # Streamlit UI
├── requirements.txt
├── .env.example
└── Dockerfile
```

---

## What I'd Build Next

- **Persistent sessions** — store FAISS index to S3 so sessions survive restarts
- **Multi-document support** — query across multiple PDFs in one session
- **Streaming responses** — stream Bedrock output token-by-token to the UI
- **Live hosted demo** — deploy to Hugging Face Spaces

---

## Related Projects

- [AgentFlow](https://github.com/Rajeshdevandla/agent-flow) — Multi-agent orchestration system with Constitutional AI safety layer
- [AI Document Intelligence Platform](https://github.com/Rajeshdevandla/ai-document-intelligence-platform) — Enterprise document processing with Java microservices + OCR

---

*Built by [Rajesh Kumar](https://rajeshdevandla.github.io) — Full Stack Java & AI Developer | Chicago, IL*
