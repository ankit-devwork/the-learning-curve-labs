# GenAI Document Assistant (Multi‑Agent RAG System)

Production-oriented document intelligence capstone built on FastAPI, LangGraph, ChromaDB, LiteLLM, and **pycorekit**.

## Features

- Multi-format ingestion: PDF, TXT, CSV, XLSX, JSON, YAML
- Multi-agent RAG pipeline with planner, selector, retriever, reasoning, and response agents
- True HITL pause when document selection is ambiguous
- Chunk-level hallucination detection and safety guardrails
- Query answer caching (in-memory by default, Redis optional)
- Observability with correlation IDs, spans, and sanitized trace payloads
- Streamlit UI for upload, chat, HITL, and observability inspection

## Project structure

```text
genai-doc-assistant-capstone/
├── app/
│   ├── agents/           # LangGraph agent nodes
│   ├── api/              # FastAPI routes
│   ├── core/             # RAG, graphs, settings, guardrails
│   ├── schema/           # Pydantic config models
│   └── service/          # ChromaDB, cache, embeddings
├── docs/
│   ├── ARCHITECTURE.md
│   ├── ARCHITECTURE_DIAGRAM.md
│   ├── CONFIGURATION.md
│   ├── DOCKER.md
│   └── EC2.md
├── front-end/streamlit/  # Streamlit UI
├── tests/
├── config.yaml
├── .env.example
├── .env.ecr.example
├── docker-compose.yml
├── docker-compose.ecr.yml
├── docker-compose.dev.yml
├── Dockerfile
└── main.py
```

## Quick start (Docker)

```bash
# From repo root
cp genai-doc-assistant-capstone/.env.example genai-doc-assistant-capstone/.env
# Edit .env and set GROQ_API_KEY

docker compose -f genai-doc-assistant-capstone/docker-compose.yml up --build
```

Or from the capstone directory:

```bash
cd genai-doc-assistant-capstone
cp .env.example .env
docker compose up --build
```

**Dev mode** (hot reload, source mounts):

```bash
docker compose -f genai-doc-assistant-capstone/docker-compose.yml -f genai-doc-assistant-capstone/docker-compose.dev.yml up --build
```

## Deploy to AWS EC2 (ECR)

Push both images to ECR, then run `docker-compose.ecr.yml` on an EC2 instance:

```bash
ECR_REGISTRY=123456789012.dkr.ecr.us-east-1.amazonaws.com \
./genai-doc-assistant-capstone/scripts/push-ecr.sh
```

See [docs/EC2.md](docs/EC2.md) for EC2 setup, IAM, and compose commands.

**Local demo only?** See [docs/FREE_DEPLOY.md](docs/FREE_DEPLOY.md) for local Docker + optional tunnel.

See [docs/DOCKER.md](docs/DOCKER.md) for full Docker documentation.

| Service | URL |
|---------|-----|
| FastAPI | http://localhost:8000 |
| API docs | http://localhost:8000/docs |
| Health | http://localhost:8000/health |
| Readiness | http://localhost:8000/ready |
| Streamlit | http://localhost:8501 |

## Local development

```bash
cd genai-doc-assistant-capstone
cp .env.example .env

pip install -e ../pycorekit
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu

uvicorn main:app --reload
```

Streamlit UI:

```bash
cd front-end/streamlit
BACKEND_URL=http://127.0.0.1:8000 streamlit run chat.py
```

## Configuration

Base settings live in `config.yaml`. Secrets and runtime overrides use `.env`.

See [docs/CONFIGURATION.md](docs/CONFIGURATION.md) for override conventions.

Example overrides:

```bash
APP_RAG__CHUNK_SIZE=500
APP_FILE_UPLOAD__MAX_FILE_SIZE_MB=25
APP_CACHE__TTL_SECONDS=7200
```

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness check |
| GET | `/ready` | Readiness check (Chroma, embeddings, API key) |
| POST | `/upload-and-ingest` | Upload and ingest a document |
| GET | `/documents` | List ingested documents |
| POST | `/ask-question` | Run query graph (with HITL pause) |
| POST | `/choose-document` | Resume answer graph after HITL |
| GET | `/observability` | Observability smoke test |

## Agent flow

### Ask question

1. Planner creates an execution plan using the LLM
2. Document selector scores documents and may trigger HITL
3. If unambiguous: retriever → reasoning → response
4. If ambiguous: pipeline pauses and returns candidate documents

### Choose document (HITL)

1. Client sends selected `doc_id`
2. Answer graph runs: retriever → reasoning → response

## Testing

```bash
pytest pycorekit/tests -q
cd genai-doc-assistant-capstone && pytest tests -q
```

## Recent improvements

- HITL graph pause (no wasted LLM calls on ambiguous selection)
- Resume graph for `/choose-document`
- LLM-powered planner agent
- Chunk-level hallucination detection
- Query cache with in-memory or Redis backend
- Readiness endpoint and improved health checks
- Duplicate upload detection by `file_hash`
- UUID-prefixed stored filenames
- Missing dependency fixes (`scikit-learn`, `pyyaml`, `openpyxl`)

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Architecture diagrams](docs/ARCHITECTURE_DIAGRAM.md)
- [Configuration](docs/CONFIGURATION.md)
- [Docker guide](docs/DOCKER.md)
- [EC2 / ECR deployment](docs/EC2.md)
- [Deployment options](docs/FREE_DEPLOY.md)
- [pycorekit README](../pycorekit/README.md)
