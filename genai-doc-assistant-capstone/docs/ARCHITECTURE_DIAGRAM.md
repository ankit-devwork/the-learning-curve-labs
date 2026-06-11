# Architecture Diagrams

Visual reference for the **GenAI Document Assistant** capstone: components, data flows, agent graphs, and deployment topology.

Related docs:
- [ARCHITECTURE.md](ARCHITECTURE.md) — narrative architecture overview
- [CONFIGURATION.md](CONFIGURATION.md) — config and `.env` overrides
- [DOCKER.md](DOCKER.md) — local Docker Compose guide
- [RENDER.md](RENDER.md) — cloud deployment on Render

---

## 1. High-level system overview

```mermaid
flowchart TB
    subgraph User["User"]
        Browser["Browser"]
    end

    subgraph Frontend["Streamlit UI :8501"]
        Chat["chat.py"]
        ObsDash["observability_dashboard.py"]
        APIClient["api_client.py"]
    end

    subgraph Backend["FastAPI Backend :8000"]
        Main["main.py"]
        Middleware["RequestTracingMiddleware"]
        Routes["API Routes"]
        Agents["LangGraph Agents"]
        Core["RAG / LLM / Guardrails"]
        Services["db_connection + query_cache"]
    end

    subgraph PyCoreKit["pycorekit (shared toolkit)"]
        Logging["core_logging"]
        Tracing["tracing + observability"]
        Config["ConfigLoader + env overrides"]
        Cache["CacheService + MemoryCache"]
        Exceptions["AppException handlers"]
    end

    subgraph Storage["Local persistence"]
        Uploads["data/uploads/"]
        Chroma["ChromaDB vector_store/"]
        Logs["logs/"]
    end

    subgraph External["External services"]
        Groq["Groq API via LiteLLM"]
        Langfuse["Langfuse optional"]
        LangSmith["LangSmith optional"]
        Redis["Redis optional"]
    end

    subgraph Config["Configuration"]
        YAML["config.yaml"]
        ENV[".env"]
    end

    Browser --> Chat
    Chat --> APIClient
    APIClient -->|HTTP REST| Routes

    Main --> Middleware --> Routes
    Routes --> Core
    Routes --> Agents
    Routes --> Services
    Agents --> Core
    Services --> Chroma

    Main --> PyCoreKit
    Routes --> PyCoreKit
    Agents --> PyCoreKit

    Core --> Groq
    Tracing --> Langfuse
    Tracing --> LangSmith
    Services --> Uploads
    Services --> Logs
    Cache --> Redis

    YAML --> Config
    ENV --> Config
    Config --> Main
```

---

## 2. Backend API layer

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Liveness — API is up |
| `/ready` | GET | Readiness — Chroma, embeddings, API key |
| `/upload-and-ingest` | POST | Upload + full ingestion pipeline |
| `/documents` | GET | List ingested documents |
| `/ask-question` | POST | Run query graph (may trigger HITL) |
| `/choose-document` | POST | Resume after user picks a document |
| `/observability` | GET | Observability smoke test |

Mutating and query routes use `@with_observability` and return a sanitized `observability` payload in JSON responses.

---

## 3. Configuration flow

```mermaid
flowchart LR
    YAML["config.yaml base settings"]
    ENV[".env secrets + overrides"]
    Loader["pycorekit ConfigLoader"]
    Settings["AppConfig settings.py"]
    App["FastAPI + Agents"]

    YAML --> Loader
    ENV -->|"python-dotenv"| Loader
    ENV -->|"APP_* overrides"| Loader
    Loader --> Settings --> App

    ENV -->|"GROQ_API_KEY"| LiteLLM["LiteLLM / Groq"]
```

**Usage:**
- `config.yaml` — chunk size, models, paths, cache TTL
- `.env` — API keys and runtime overrides (e.g. `APP_RAG__CHUNK_SIZE=500`)

---

## 4. Document ingestion pipeline

**Trigger:** `POST /upload-and-ingest` (Streamlit sidebar or API client)

```mermaid
flowchart TD
    Start(["User uploads file"]) --> Validate["Validate type + size"]
    Validate --> DupCheck["SHA-256 duplicate check Chroma metadata"]
    DupCheck -->|Duplicate| Skip["Return duplicate=true"]
    DupCheck -->|New| Save["Save to data/uploads uuid_filename"]
    Save --> Parse["Parse PDF/TXT/CSV/XLSX/JSON/YAML"]
    Parse --> Clean["basic_clean"]
    Clean --> Chunk["chunk_text sliding/recursive/hybrid"]
    Chunk --> Quality["filter_low_quality"]
    Quality --> Dedupe["dedupe_exact + semantic dedupe"]
    Dedupe --> Summary["LLM summary Groq via LiteLLM"]
    Summary --> Embed["SentenceTransformer all-mpnet-base-v2"]
    Embed --> Store["ChromaDB documents collection"]
    Store --> Invalidate["Invalidate query cache"]
    Invalidate --> Done(["Return doc_id summary observability"])
```

**Chroma metadata per chunk:** `doc_id`, `title`, `summary`, `filename`, `file_hash`, `chunk_index`

---

## 5. Query flow with HITL

**Trigger:** `POST /ask-question` with `{ question, thread_id }`

```mermaid
flowchart TD
    QStart(["POST /ask-question"]) --> Cache{"Query cache hit?"}
    Cache -->|Yes| CReturn["Return cached answer"]
    Cache -->|No| History["Load chat history Chroma"]
    History --> QGraph["query_graph.ainvoke"]

    subgraph QueryGraph["Query Graph LangGraph"]
        P["1 Planner Agent LLM plan"]
        DS["2 Document Selector embed + score"]
        P --> DS
        DS --> Ambiguous{"Ambiguous? needs_user_choice"}
        Ambiguous -->|Yes| HITL_END(["Graph END"])
        Ambiguous -->|No| R["3 Retriever Chroma top-k"]
        R --> RE["4 Reasoning Agent LLM"]
        RE --> RS["5 Response Agent + guardrails"]
    end

    QGraph --> P
    HITL_END --> HITLResp["Return candidate_documents"]
    RS --> SaveHist["Save chat to Chroma"]
    SaveHist --> CacheSet["Store in query cache"]
    CacheSet --> Answer(["Return answer + confidence"])
```

---

## 6. HITL resume flow

**Trigger:** User selects a document → `POST /choose-document`

```mermaid
flowchart TD
    HStart(["POST /choose-document"]) --> ACache{"Cache hit?"}
    ACache -->|Yes| AReturn["Return cached answer"]
    ACache -->|No| AHist["Load chat history"]

    subgraph AnswerGraph["Answer Graph resume only"]
        AR["Retriever scoped to selected_doc_id"]
        ARE["Reasoning Agent"]
        ARS["Response Agent + guardrails"]
        AR --> ARE --> ARS
    end

    AHist --> AR
    ARS --> ASave["Save chat history"]
    ASave --> ACacheSet["Cache answer"]
    ACacheSet --> ADone(["Return final answer"])
```

The answer graph skips planner and document selector to avoid redundant LLM calls.

---

## 7. LangGraph agent responsibilities

```mermaid
flowchart LR
    subgraph Agents
        Planner["Planner question + chat_history to plan"]
        Selector["Document Selector scores docs may trigger HITL"]
        Retriever["Retriever Chroma similarity search"]
        Reasoning["Reasoning LLM context summary"]
        Response["Response final answer + guardrails"]
    end

    Planner --> Selector --> Retriever --> Reasoning --> Response
```

| Agent | Input | Output |
|-------|--------|--------|
| Planner | `question`, `chat_history` | `steps` (LLM plan) |
| Document Selector | `question` | `selected_doc_id` or `candidate_docs` |
| Retriever | `question`, `selected_doc_id?` | `retrieved_chunks` |
| Reasoning | chunks + history | `reasoning_summary` |
| Response | reasoning + chunks | `final_answer`, `confidence`, `hallucinated` |

---

## 8. ChromaDB collections

```mermaid
erDiagram
    DOCUMENTS {
        string id
        string document
        vector embedding
        string doc_id
        string title
        string summary
        string filename
        string file_hash
        int chunk_index
    }

    CHAT_HISTORY {
        string id
        string document
        vector dummy_embedding
        string thread_id
        string role
        string content
        string timestamp
    }
```

| Collection | Used for |
|------------|----------|
| `documents` | Chunk vectors + document metadata |
| `chat_history` | Per-`thread_id` conversation (dummy embeddings) |

---

## 9. pycorekit cross-cutting concerns

```mermaid
flowchart TB
    Request["Incoming HTTP Request"] --> MW["RequestTracingMiddleware"]
    MW --> CID["correlation_id x-correlation-id"]
    MW --> Trace["init_empty_trace"]
    MW --> Route["API Route Handler"]

    Route --> Decorator["@with_observability"]
    Decorator --> Span["start_trace spans"]
    Span --> LF["Langfuse optional"]
    Span --> LS["LangSmith optional"]

    Route --> Log["core_logging"]
    Route --> Exc["AppException handlers"]

    Decorator --> Sanitize["sanitize_observability"]
    Sanitize --> Response["JSON + observability payload"]
```

---

## 10. Streamlit UI usage map

```mermaid
flowchart LR
    subgraph Streamlit["Streamlit :8501"]
        Upload["Sidebar Upload file"]
        DocList["Sidebar List documents"]
        ChatUI["Main Chat input"]
        HITL["HITL radio + Submit"]
        Obs["Observability expander"]
    end

    Upload -->|POST /upload-and-ingest| API
    DocList -->|GET /documents| API
    ChatUI -->|POST /ask-question| API
    HITL -->|POST /choose-document| API
    Obs -->|observability in response| API

    API["FastAPI :8000"]
```

---

## 11. Project deployment artifacts

All capstone deployment files live inside `genai-doc-assistant-capstone/` (not the monorepo root):

```text
genai-doc-assistant-capstone/
├── docker-compose.yml       # Local stack (project: genai-doc-assistant-capstone)
├── docker-compose.dev.yml   # Dev overrides (hot reload)
├── render.yaml              # Render Blueprint (cloud deploy)
├── Dockerfile               # Backend image (build context = monorepo root)
├── front-end/streamlit/Dockerfile
├── .env.example
└── docs/
    ├── DOCKER.md
    └── RENDER.md
```

Docker and Render builds still use the **monorepo root** as context (`..` or `.`) because the backend image copies `pycorekit/`.

---

## 12. Docker deployment topology (local)

```mermaid
flowchart TB
    subgraph Host["Host machine"]
        DC["docker compose -f genai-doc-assistant-capstone/docker-compose.yml up"]
    end

    subgraph Network["genai-net"]
        BE["genai_backend :8000"]
        ST["genai_streamlit :8501"]
    end

    subgraph Volumes["Named volumes (genai-doc-assistant-capstone_*)"]
        V1["genai-uploads"]
        V2["genai-vector-store"]
        V3["genai-logs"]
    end

    ENV["genai-doc-assistant-capstone/.env"] --> BE
    DC --> BE
    DC --> ST
    ST -->|"BACKEND_URL=http://backend:8000"| BE
    BE --> V1
    BE --> V2
    BE --> V3
```

---

## 13. Render deployment topology (cloud)

```mermaid
flowchart TB
    subgraph Render["Render.com"]
        Blueprint["genai-doc-assistant-capstone/render.yaml"]
        BE["genai-backend Web Service"]
        ST["genai-streamlit Web Service"]
        Disk["Persistent disk /app/data"]
    end

    subgraph External["External"]
        Groq["Groq API"]
        User["User browser"]
    end

    Blueprint --> BE
    Blueprint --> ST
    User --> ST
    ST -->|"BACKEND_HOSTPORT private network"| BE
    BE --> Disk
    BE --> Groq
```

---

## 14. End-to-end user journey (sequence)

```mermaid
sequenceDiagram
    actor User
    participant UI as Streamlit
    participant API as FastAPI
    participant Graph as LangGraph
    participant Chroma as ChromaDB
    participant LLM as Groq via LiteLLM

    User->>UI: Upload PDF
    UI->>API: POST /upload-and-ingest
    API->>Chroma: Store chunks + embeddings
    API-->>UI: doc_id, summary

    User->>UI: Ask question
    UI->>API: POST /ask-question
    API->>Chroma: Load chat history
    API->>Graph: query_graph
    Graph->>LLM: Planner plan
    Graph->>Chroma: Document selector scores

    alt Ambiguous documents
        Graph-->>API: needs_user_choice + candidates
        API-->>UI: Show HITL picker
        User->>UI: Select document
        UI->>API: POST /choose-document
        API->>Graph: answer_graph
    else Clear winner
        Graph->>Chroma: Retrieve chunks
        Graph->>LLM: Reasoning + Response
    end

    Graph->>LLM: Generate answer
    API->>Chroma: Save chat messages
    API-->>UI: answer + observability
    UI-->>User: Display answer
```

---

## 15. Component summary

| Component | Technology | Role |
|-----------|------------|------|
| Streamlit UI | Python | User interface |
| FastAPI | Python | REST API |
| LangGraph | Python | Multi-agent orchestration |
| ChromaDB | Embedded DB | Vectors + chat history |
| SentenceTransformers | Local model | Embeddings (768-dim) |
| LiteLLM | Python | LLM routing to Groq |
| pycorekit | Internal library | Logging, tracing, config, cache |
| config.yaml + .env | YAML + dotenv | Configuration |
| MemoryCache / Redis | In-process / optional | Query answer cache |

---

## Viewing these diagrams

- **GitHub** renders Mermaid blocks in this file automatically.
- **VS Code** — install a Mermaid preview extension.
- **Export to PNG/SVG** — use [mermaid.live](https://mermaid.live) or the Mermaid CLI.
