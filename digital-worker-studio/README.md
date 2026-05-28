# Digital Worker Studio

Digital Worker Studio is a modular AI workflow automation platform for building, running, and monitoring “digital workers.” It combines a visual workflow builder, multi-agent execution engine, cost-aware model routing, semantic document intelligence, and deep observability.

---

## 🚀 Project Overview

Digital Worker Studio enables teams to:
- build AI workflows using a visual canvas,
- ingest documents and extract structured knowledge,
- execute multi-agent reasoning pipelines,
- route tasks through cost-aware SLM/LLM selection,
- inspect runtime traces, metrics, and logs.

This repository contains the core backend services, APIs, and documentation for the platform.

---

## 🧩 Key Features

- Visual workflow builder support via React + React Flow
- Multi-agent execution engine with planner/worker/critic loops
- Cost-aware model routing using LiteLLM
- PDF document ingestion, chunking, and embedding
- Graph-backed query engine with LangGraph
- Redis-backed cache and performance optimization
- Postgres persistence for documents and workflow state
- Neo4j knowledge graph sync
- Observability via LangSmith / Langfuse-compatible tracing

---

## 🏗 Architecture

### Core Components

- **Frontend**: React-based visual workflow builder and dashboards
- **Backend**: FastAPI HTTP application and service layer
- **Workers**: Background execution processes for document and agent tasks
- **Redis**: Cache and queue state engine
- **Postgres**: Durable storage for metadata and embeddings
- **Neo4j**: Graph database for knowledge synchronization
- **LiteLLM**: Unified LLM gateway for inference routing
- **LangSmith / Langfuse**: Tracing and observability telemetry

### Execution Flow

1. User submits a request through UI or API.
2. FastAPI routes the request and validates configuration.
3. The backend persists document or query state.
4. Worker processes execute long-running AI operations.
5. LiteLLM performs model inference and routing.
6. Results are stored and returned to the client.
7. Observability and tracing capture each stage of execution.

---

## 📁 Repository Structure

```
.digital-worker-studio/
├─ backend/
│  ├─ app/
│  │  ├─ api/                 # FastAPI route definitions
│  │  │  ├─ routes/
│  │  │  │  ├─ document_ingestion.py
│  │  │  │  ├─ query_engine.py
│  │  │  │  ├─ observability.py
│  │  │  ├─ __init__.py
│  │  ├─ agents/              # LangGraph workflows and agent logic
│  │  ├─ core/                # Configuration, DB, Redis, graph services
│  │  ├─ models/              # ORM models for document storage
│  │  ├─ observability/       # Logging, middleware, decorators
│  │  ├─ schemas/             # Pydantic request/response schemas
│  │  ├─ services/            # Cache, graph sync, and processing services
│  │  ├─ utils/               # Utility helpers
│  │  ├─ main.py              # FastAPI application entry point
│  ├─ workers/                # Worker process code
│
├─ frontend/                  # React frontend source
│  ├─ src/
│
├─ docs/                      # Architecture and roadmap docs
│  ├─ ARCHITECTURE.md
│  ├─ TECH_STACK.md
│  ├─ PHASE_WISE_FEATURES.md
│  ├─ PROJECT_PLAN.md
│
├─ config.properties          # Application configuration
├─ create_missing_folders.py  # Utility to create folder tree from docs
├─ requirements.txt           # Python dependencies
├─ README.md
```

---

## 🛠 Prerequisites

- Python 3.11+
- Redis
- PostgreSQL
- Neo4j
- Optional: Tesseract OCR for scanned PDFs
- Optional: LangSmith/Langfuse credentials
- Optional: LiteLLM provider credentials (e.g., Groq API key)

---

## ⚙️ Configuration

### `config.properties`

The backend loads runtime settings from `config.properties`.
Important values include:

- `BASE_GENERATION_MODEL`
- `text_embedding_model`
- `chunk_size`
- `chunk_overlap`
- `redis.host`, `redis.port`, `redis.db`, `redis.password`
- `db.host`, `db.port`, `db.user`, `db.password`, `db.name`
- `neo4j.uri`, `neo4j.user`, `neo4j.password`
- `storage.local_dir`, `storage.max_file_size_mb`

### Environment Variables

Set provider secrets before launching the app:

- `GROQ_API_KEY`
- Additional model provider keys as required

---

## 🚀 Installation

From `digital-worker-studio/`:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## ▶️ Running the Backend

From `digital-worker-studio/backend`:

```bash
cd digital-worker-studio/backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Then open the FastAPI docs:

```text
http://127.0.0.1:8000/docs
```

---

## 📡 API Endpoints

### Document Ingestion

- `POST /api/worker/document/upload`
  - Uploads a PDF file
  - Extracts text and creates text chunks
  - Generates embeddings
  - Persists document and chunk data
  - Runs Groq extraction via LiteLLM

### Graph RAG Search

- `POST /api/worker/query/ask`
  - Body:
    - `question`: natural language query
    - `thread_id`: session identifier
  - Executes Graph-RAG query loop
  - Uses Redis cache for repeat queries
  - Returns a structured answer

### Observability & Diagnostics

- `GET /api/monitor/observability-check`
  - Runs a LiteLLM observability test
- `GET /api/monitor/db-check`
  - Validates PostgreSQL connectivity
- `GET /api/monitor/redis-check`
  - Validates Redis connectivity
- `GET /api/monitor/neo4j-check`
  - Validates Neo4j connectivity

---

## 🔧 Backend Behavior

### `backend/app/main.py`

- Creates FastAPI app with lifespan management
- Initializes Postgres schema automatically
- Configures LangGraph Postgres checkpointer
- Initializes Redis cache service
- Registers API routes and exception handlers

### `backend/app/api/routes/document_ingestion.py`

- Handles PDF-only ingestion
- Supports pypdf text extraction
- Optional OCR fallback with `pdf2image`/`pytesseract`
- Uses sliding window chunking and embedding
- Persists chunk vectors into Postgres
- Runs Groq extraction for structured insights

### `backend/app/api/routes/query_engine.py`

- Accepts question + thread session
- Looks up Redis cache before re-running graph logic
- Uses LangGraph executor for cognitive search
- Saves chat history into Postgres checkpoint state
- Returns final answer payload

### `backend/app/api/routes/observability.py`

- Exposes readiness and health checks
- Uses observability decorators for tracing
- Validates LiteLLM connectivity in-flight

---

## 📘 Development Notes

- Configuration uses a singleton loader in `backend/app/core/load_property.py`
- Document ingestion is PDF-first and uses local embeddings
- Query engine is stateful and thread-aware via `thread_id`
- Observability middleware propagates trace IDs across workers
- Frontend sources are located in `frontend/`, but may require a separate React setup

### Utility

- `create_missing_folders.py` generates the folder layout from `docs/FOLDER_STRUCTURE.md`

---

## 🧪 Local Setup Checklist

1. Confirm `config.properties` points to local Redis/Postgres/Neo4j.
2. Set `GROQ_API_KEY` and other provider keys in your shell.
3. Start infrastructure services before launching the backend.
4. Open `/docs` once the backend is running.

---

## 📈 Roadmap

See `docs/PHASE_WISE_FEATURES.md` for the planned evolution of the product:

- Phase 1: MVP workflow engine + LiteLLM
- Phase 2: Visual workflow builder
- Phase 3: Multi-agent execution and fallback logic
- Phase 4: Integrations with Gmail, Notion, Slack, Drive, GitHub
- Phase 5: AI Operator mode
- Phase 6: Marketplace and templates

---

## 📚 Documentation

Additional docs available in `docs/`:
- `ARCHITECTURE.md`
- `TECH_STACK.md`
- `PROJECT_PLAN.md`
- `PHASE_WISE_FEATURES.md`

---

## 📌 Notes

- The backend is built for Python 3.11+.
- Start the app from `digital-worker-studio/backend`.
- Ensure required service endpoints are available before startup.
- The React frontend may require a separate install step.
