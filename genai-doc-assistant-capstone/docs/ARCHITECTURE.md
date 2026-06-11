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
