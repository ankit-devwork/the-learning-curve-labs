# GenAI Document Assistant (Multi‑Agent RAG System)

## Overview

GenAI Document Assistant is a multi‑agent Retrieval‑Augmented Generation (RAG) system designed for production‑grade document intelligence workflows.

It enables you to:

- Upload and ingest documents  
- Parse, clean, chunk, dedupe, and embed them  
- Ask natural‑language questions  
- Use a multi‑agent reasoning pipeline  
- Handle ambiguous document selection via HITL  
- Chat with your documents through a Streamlit UI  

The system integrates:

- FastAPI backend  
- Streamlit frontend  
- ChromaDB vector store  
- LangGraph multi‑agent orchestration  
- LiteLLM for LLM calls  
- pycorekit for config, logging, tracing, observability, uploading, and utilities  

---

## Key Features

### 1. Unified Upload + Ingest Pipeline

Supported file types:

- PDF  
- TXT  
- CSV  
- XLSX  
- JSON  
- YAML  

The ingestion pipeline performs:

- File validation  
- Safe local upload to configured `settings.paths.upload_dir`  
- Filename sanitization to prevent path traversal  
- Parsing  
- Cleaning  
- Chunking (sliding / recursive / hybrid)  
- Quality scoring  
- Exact + semantic deduplication  
- LLM‑powered document summary  
- Embedding + storage in ChromaDB  

**Endpoint:** `POST /upload-and-ingest`

---

### 2. Multi‑Agent RAG Querying

The system uses:

- Planner Agent  
- Document Selector Agent  
- Retriever Agent  
- Reasoning Agent  
- Response Agent  

**Endpoint:** `POST /ask-question`

---

### 3. Human‑in‑the‑Loop (HITL) Document Selection

If multiple documents match your question:

- Backend returns candidate documents  
- User selects one  
- Pipeline resumes from retriever  

**Endpoint:** `POST /choose-document`

---

### 4. Document Listing

List all ingested documents with:

- doc_id  
- title  
- summary  

**Endpoint:** `GET /documents`

---

### 5. Observability and Error Handling

The backend uses `pycorekit` to provide:

- request trace initialization via middleware  
- automatic injection of sanitized observability payloads into JSON responses  
- `AppException` and `FileException` for structured error responses  
- correlation IDs returned in response headers (`x-correlation-id`)  

This means route handlers do not need to manually serialize trace data.

---

### 6. Streamlit Frontend

Provides:

- Document upload  
- Document list  
- Chat interface  
- HITL document selection  
- Response visualization  

---

### 7. Docker‑First Architecture

Run everything with:

```
docker-compose up --build
```

---

## Project Structure

```
the-learning-curve-labs/
│
├── pycorekit/                     # Shared utilities (mounted into backend)
│
└── genai-doc-assistant-capstone/
    ├── app/
    │   ├── core/                  # RAG logic, settings, LLM, chunking
    │   ├── service/               # DB + embedding services
    │   ├── api/                   # FastAPI routes
    │   └── main.py                # FastAPI entrypoint
    │
    ├── front-end/
    │   └── streamlit/
    │       ├── chat.py            # Streamlit UI
    │       └── api_client.py      # Backend client wrapper
    │
    ├── Dockerfile                 # Backend Dockerfile
    ├── requirements.txt
    └── config.yaml                # System configuration
│
├── docker-compose.yml             # Root-level compose file
└── README.md
```

---

## How to Run the Project (Docker)

### 1. Clone the repository

```
git clone <your-repo-url>
cd the-learning-curve-labs
```

### 2. Create a `.env` file (optional)

```
OPENAI_API_KEY=your_key
GROQ_API_KEY=your_key
```

### 3. Start the system

```
docker-compose up --build
```

### 4. Access the services

| Service | URL |
|--------|-----|
| FastAPI Backend | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |
| Streamlit UI | http://localhost:8501 |

---

## How to Run Locally (Without Docker)

### 1. Install backend dependencies

```
cd genai-doc-assistant-capstone
pip install -r requirements.txt
pip install ../pycorekit
```

### 2. Start FastAPI

```
uvicorn main:app --reload
```

### 3. Start Streamlit

```
cd front-end/streamlit
streamlit run chat.py
```

---

## Multi‑Agent Pipeline Overview

### 1. Upload + Ingest
- File saved  
- Parsed  
- Cleaned  
- Chunked  
- Deduped  
- Summarized  
- Embedded  
- Stored in ChromaDB  

### 2. Ask Question
- Planner decides steps  
- Document selector picks best doc  
- If ambiguous → HITL  
- Retriever fetches chunks  
- Reasoning agent synthesizes  
- Response agent answers  

### 3. Choose Document (HITL)
- User selects doc  
- Pipeline resumes  
- Final answer returned  

---

## API Usage Examples

### Upload a document

```bash
curl -X POST "http://localhost:8000/upload-and-ingest" \
  -F "file=@sample.pdf"
```

### Ask a question

```bash
curl -X POST "http://localhost:8000/ask-question" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this document about?", "thread_id": "123"}'
```

### Choose a document

```bash
curl -X POST "http://localhost:8000/choose-document" \
  -H "Content-Type: application/json" \
  -d '{"thread_id": "123", "question": "What is this?", "selected_doc_id": "abc"}'
```

### List documents

```bash
curl http://localhost:8000/documents
```

---

## Tech Stack

- FastAPI  
- Streamlit  
- ChromaDB  
- LangGraph  
- LiteLLM  
- pycorekit  
- Docker Compose  

---

## Future Enhancements

- Agent trace visualization  
- Document preview + chunk viewer  
- Multi‑document reasoning  
- Embedding dashboard  
- Multi‑tenant support  

---

## Contributing

PRs are welcome.  
Open an issue to propose features or report bugs.

