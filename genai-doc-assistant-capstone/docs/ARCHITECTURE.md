# Architecture

> For visual diagrams (Mermaid), see [ARCHITECTURE_DIAGRAM.md](ARCHITECTURE_DIAGRAM.md).

## Components

| Layer | Technology | Responsibility |
|-------|------------|----------------|
| API | FastAPI | HTTP routes, validation, error handling |
| Agents | LangGraph | Multi-step RAG orchestration |
| Vector store | ChromaDB | Embeddings, document chunks, chat history |
| LLM gateway | LiteLLM | Model routing (Groq/OpenAI) |
| Toolkit | pycorekit | Logging, tracing, config, cache, exceptions |
| UI | Streamlit | Upload, chat, HITL selection, observability |

## Agent graphs

### Query graph (`query_graph`)

Used by `POST /ask-question`.

```text
planner -> document_selector -> [HITL END | retriever -> reasoning -> response]
```

When multiple documents match within the ambiguity margin, the graph stops after `document_selector` and returns candidate documents to the client.

### Answer graph (`answer_graph`)

Used by `POST /choose-document` after HITL selection.

```text
retriever -> reasoning -> response
```

This avoids re-running planner and document selector.

## Caching

- Default: in-memory cache via `pycorekit.cache.MemoryCache`
- Optional: Redis when `REDIS_HOST` is set
- Cache is invalidated after successful document ingestion

## Observability

- `RequestTracingMiddleware` initializes per-request traces
- `@with_observability` injects sanitized trace payloads into JSON responses
- Langfuse and LangSmith are optional and lazily initialized

## Deployment

Deployment artifacts live inside the capstone project folder (this monorepo contains multiple projects):

| Artifact | Path | Purpose |
|----------|------|---------|
| Docker Compose | `docker-compose.yml` | Local backend + Streamlit stack |
| Docker Compose (dev) | `docker-compose.dev.yml` | Hot-reload overrides |
| Docker Compose (ECR) | `docker-compose.ecr.yml` | EC2 pull-and-run from ECR |
| ECR push script | `scripts/push-ecr.sh` | Build, tag, push both images |
| Backend Dockerfile | `Dockerfile` | API image (build context = monorepo root) |
| Streamlit Dockerfile | `front-end/streamlit/Dockerfile` | UI image |

Guides:
- Local: [DOCKER.md](DOCKER.md)
- Cloud: [EC2.md](EC2.md)
- Options: [FREE_DEPLOY.md](FREE_DEPLOY.md)
- Diagrams: [ARCHITECTURE_DIAGRAM.md](ARCHITECTURE_DIAGRAM.md) (sections 11–13)
