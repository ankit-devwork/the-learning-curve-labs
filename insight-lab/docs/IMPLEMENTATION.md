# InsightLab — Implementation Progress

Living document updated as features land. See [ARCHITECTURE.md](ARCHITECTURE.md) for system design.

## Phase 1 — Foundation

| Step | Feature | Status | Docs / endpoints |
|------|---------|--------|------------------|
| 1.0 | Repo scaffold, MIT, Docker Redis/Neo4j | Done | [README](../README.md) |
| 1.1 | Conda env (`insightlab`) | Done | [backend/README](../backend/README.md) |
| 1.2 | FastAPI health + `/ready` checks | Done | `GET /health`, `GET /ready` |
| 1.3 | Supabase schema + Storage bucket | Done | [supabase/migrations](../supabase/migrations/001_initial.sql) |
| 1.4 | Next.js auth (email + Google) | Done | [frontend/README](../frontend/README.md) |
| **1.5** | **pycorekit + JWT + `GET /me`** | **Done** | `GET /me`, dev-only Backend connection card |
| **1.6** | **File upload API** | **Done** | `POST /upload`, `GET /documents`, dashboard upload UI |
| **1.7** | **Document summary + chat** | **Done** | `POST /process`, `GET /summary`, `POST /ask`, Redis cache |
| **1.7b** | **pgvector embeddings (RAG)** | **Done** | fastembed + `match_document_chunks` RPC, vector retrieval |
| 1.8 | Excel charts pipeline | Planned | `POST /excel/analyze` |
| 1.9 | Quiz generator | Planned | `POST /quiz/generate` |

---

## Step 1.5 — pycorekit + backend JWT (implemented)

### Backend

| Component | File | Purpose |
|-----------|------|---------|
| Logging | `pycorekit.core_logging` | Structured logs in `backend/logs/` |
| Tracing middleware | `pycorekit.tracing.middleware` | Correlation ID on every request |
| Route observability | `@with_observability` | Spans on `/health`, `/ready`, `/me` |
| Exception handlers | `pycorekit.exceptions.handlers` | Consistent JSON errors |
| JWT decode | `app/core/auth.py` | Verify Supabase access token |
| Auth dependency | `app/core/deps.py` | `Depends(get_current_user)` |
| Profile route | `app/api/routes/me.py` | `GET /me` |

### Frontend

| Component | File | Purpose |
|-----------|------|---------|
| Backend probe | `components/auth/backend-me-card.tsx` | Calls `GET /me` with session JWT |
| Dashboard | `app/dashboard/page.tsx` | Shows backend-verified user |

### Auth flow (frontend → backend)

```mermaid
sequenceDiagram
  participant Dashboard as Next.js Dashboard
  participant Supabase as Supabase Session
  participant API as FastAPI /me
  participant PyCore as pycorekit

  Dashboard->>Supabase: getSession()
  Supabase-->>Dashboard: access_token (JWT)
  Dashboard->>API: Authorization: Bearer JWT
  API->>PyCore: RequestTracingMiddleware
  API->>API: jwt.decode (HS256 secret or ES256 JWKS)
  API-->>Dashboard: user_id, email, correlation_id
```

### Install pycorekit

```powershell
conda activate insightlab
cd backend
pip install -e D:\Mine\Learining\GenAI\python\the-learning-curve-labs\pycorekit
```

### Verify

1. Start backend: `uvicorn app.main:app --reload`
2. Start frontend: `npm run dev`
3. Sign in → dashboard → **Backend connection** card shows user ID + correlation ID

### Troubleshooting `/me` returns 401

| Cause | Fix |
|-------|-----|
| **New Supabase signing keys (ES256)** | Ensure `SUPABASE_URL` is set in `backend/.env` — backend auto-fetches JWKS |
| **Legacy JWT secret (HS256)** | Ensure `SUPABASE_JWT_SECRET` matches **Settings → API → JWT Secret** |
| Wrong secret pasted | Re-copy JWT Secret; no extra spaces or quotes |
| Token expired | Sign out and sign in again |

Check token algorithm in [jwt.io](https://jwt.io): header `alg` is `HS256` or `ES256`.

---

## Environment variables (Step 1.5)

### Backend (`backend/.env`)

| Variable | Required | Purpose |
|----------|----------|---------|
| `SUPABASE_URL` | Yes | JWKS fetch for ES256 tokens + readiness check |
| `SUPABASE_JWT_SECRET` | HS256 only | Verify legacy HS256 tokens on `/me` |
| `SUPABASE_SERVICE_ROLE_KEY` | Yes | Readiness check |
| `LOG_DIR` | No | Default `logs` |
| `CORS_ALLOW_ORIGINS` | No | Default `localhost:3000` |

### Frontend (`frontend/.env.local`)

| Variable | Required | Purpose |
|----------|----------|---------|
| `NEXT_PUBLIC_SUPABASE_URL` | Yes | Auth |
| `NEXT_PUBLIC_SUPABASE_ANON_KEY` | Yes | Auth |
| `NEXT_PUBLIC_API_URL` | Yes | `http://localhost:8000` for `/me` |
| `NEXT_PUBLIC_SHOW_DEV_PANEL` | No | `true` to show backend JWT debug card |

---

## Step 1.6 — File upload (implemented)

### Backend

| Component | File | Purpose |
|-----------|------|---------|
| Supabase client | `app/core/supabase_client.py` | Service-role Storage + Postgres |
| Upload service | `app/services/upload.py` | Validate, store blob, insert `documents` row |
| Routes | `app/api/routes/upload.py` | `POST /upload`, `GET /documents` |

### Frontend

| Component | File | Purpose |
|-----------|------|---------|
| API helper | `lib/api.ts` | Bearer-authenticated fetch to FastAPI |
| Upload UI | `components/documents/file-upload-card.tsx` | Choose file + list uploads |
| Dev panel gate | `components/auth/dev-backend-me-card.tsx` | Shows `/me` card only when `NEXT_PUBLIC_SHOW_DEV_PANEL=true` |

### Allowed file types

- **Excel:** `.xlsx`, `.xls`, `.csv`
- **Documents:** `.pdf`, `.txt`, `.docx`, `.doc`

Max size: 20 MB (configurable via `UPLOAD_MAX_BYTES` in backend `.env`).

### Verify

1. Ensure Supabase Storage bucket **`uploads`** exists (private)
2. `pip install -r requirements.txt` (adds `supabase`, `python-multipart`)
3. Sign in → dashboard → **Upload files** → choose a PDF or Excel file
4. File appears in the list with status `pending`

---

## Step 1.7 — Document summary + chat (implemented)

### Backend

| Component | File | Purpose |
|-----------|------|---------|
| Cache | `app/core/cache.py` | pycorekit `CacheService` + rate limits |
| Document service | `app/services/document_service.py` | Parse, chunk, summarize, ask |
| Text extraction | `app/services/document_text.py` | PDF, txt, docx parsers |
| LLM client | `app/services/llm_client.py` | LiteLLM / Groq for summary + chat |
| Routes | `app/api/routes/documents.py` | Process, summary, ask endpoints |

### Frontend

| Component | File | Purpose |
|-----------|------|---------|
| Document detail | `app/dashboard/documents/[id]/page.tsx` | Auto-process + summary + chat |
| Detail UI | `components/documents/document-detail-client.tsx` | Summary, chat, retrieval method badge |

### Database

Run `supabase/migrations/002_document_chunks.sql` — adds `summary`, `processed_at`, `document_chunks` table.

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/documents/{id}` | Document metadata + status |
| POST | `/documents/{id}/process` | Extract text, chunk, embed, summarize |
| GET | `/documents/{id}/summary` | Cached summary (Redis → Postgres) |
| POST | `/documents/{id}/ask` | RAG chat with cited chunks |

### Verify

1. Run migration `002_document_chunks.sql`
2. Set `GROQ_API_KEY` in `backend/.env`
3. Start Redis (`docker compose up -d`)
4. Upload a PDF → open document detail → auto-process → ask a question

---

## Step 1.7b — pgvector embeddings (implemented)

Semantic chunk retrieval replaces keyword-only matching when embeddings are present.

### Backend

| Component | File | Purpose |
|-----------|------|---------|
| Embeddings | `app/services/embeddings.py` | fastembed (default) or LiteLLM provider |
| Vector search | Supabase RPC | `match_document_chunks` cosine similarity |
| Config | `config.yaml` → `embeddings` | Model, dimensions, batch size, threshold |

Default model: **BAAI/bge-small-en-v1.5** (384 dimensions, local via fastembed — no extra API key).

### Database

Run `supabase/migrations/003_pgvector_embeddings.sql`:

- Enables `vector` extension
- Adds `document_chunks.embedding vector(384)` + HNSW index
- Creates `match_document_chunks(filter_document_id, query_embedding, match_count)` RPC

### Retrieval flow

```mermaid
flowchart LR
  Q[User question] --> E[Embed query]
  E --> V[Vector search RPC]
  V -->|matches above threshold| LLM[LLM answer]
  V -->|no matches| K[Keyword fallback]
  K --> LLM
```

Ask responses include `retrieval_method` (`vector` or `keyword`) and `chunk_similarities` when vector search succeeds.

### Verify

1. Run migration `003_pgvector_embeddings.sql`
2. `pip install -r requirements.txt` (adds `fastembed`)
3. **Re-process** documents uploaded before this migration (old chunks have no embeddings)
4. Ask a question — UI shows retrieval method; API returns `"retrieval_method": "vector"`

### Config overrides

```yaml
embeddings:
  provider: fastembed
  model: BAAI/bge-small-en-v1.5
  dimensions: 384
  similarity_threshold: 0.35
```

Env prefix: `APP_EMBEDDINGS__*` (e.g. `APP_EMBEDDINGS__SIMILARITY_THRESHOLD=0.4`).

---

## Next up — Step 1.8 Excel charts pipeline

- Parse uploaded spreadsheets
- Auto-generate charts and narrative insights
