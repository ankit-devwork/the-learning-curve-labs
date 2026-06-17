# FastAPI + LangGraph backend for InsightLab

## Python environment (Conda — recommended)

Requires [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda.

### First-time setup

```powershell
cd backend

conda env create -f environment.yml
conda activate insightlab

# Required: pycorekit from the-learning-curve-labs
pip install -e D:\Mine\Learining\GenAI\python\the-learning-curve-labs\pycorekit

pip install -r requirements.txt
copy .env.example .env
# Edit .env — Supabase JWT secret, Neo4j, Redis, GROQ_API_KEY
```

```bash
# macOS / Linux
cd backend
conda env create -f environment.yml
conda activate insightlab
pip install -e ../the-learning-curve-labs/pycorekit
pip install -r requirements.txt
cp .env.example .env
```

### Run API

```powershell
conda activate insightlab
uvicorn app.main:app --reload --port 8000
```

- API: http://localhost:8000  
- Docs: http://localhost:8000/docs  
- Health: http://localhost:8000/health  
- Readiness: http://localhost:8000/ready  
- **Auth test:** http://localhost:8000/me (requires Bearer JWT)

---

## pycorekit integration

InsightLab uses [pycorekit](https://github.com/ankit-devwork/the-learning-curve-labs/tree/main/pycorekit) for:

| Feature | Usage |
|---------|--------|
| Structured logging | `init_logger()` in `app/main.py` → `backend/logs/` |
| Correlation IDs | `RequestTracingMiddleware` → `x-correlation-id` header |
| Route tracing | `@with_observability` on health and auth routes |
| JSON errors | `AppException` handlers |

Protected routes use `Depends(get_current_user)` which verifies the Supabase JWT from the frontend.

### Upload settings (`config.yaml`)

Upload limits, allowed extensions, MIME types, and file signatures live in `backend/config.yaml` (not hardcoded in Python).

Override at runtime with env vars, e.g.:

```env
APP_UPLOAD__MAX_BYTES=10485760
APP_UPLOAD__STORAGE_BUCKET=uploads
```

Validation runs **before** Supabase Storage upload: extension, size (10 MB default), MIME type, and magic-byte signature checks.

### Step 1.7 — Document summary + chat

Requires:

1. Run migration `supabase/migrations/002_document_chunks.sql` in Supabase SQL Editor
2. Run migration `supabase/migrations/003_pgvector_embeddings.sql` for semantic RAG
3. `GROQ_API_KEY` in `backend/.env`
4. Redis running (`docker compose up -d`) for cache + rate limits
5. `pip install -r requirements.txt` (`fastembed`, `pypdf`, `python-docx`)

Cache keys (via pycorekit `CacheService` + Redis):

- `insightlab:summary:document:{id}`
- `insightlab:chat:document:{id}:{question_hash}`

### Embeddings (`config.yaml` → `embeddings`)

| Setting | Default | Purpose |
|---------|---------|---------|
| `provider` | `fastembed` | Local embeddings (no API key) |
| `model` | `BAAI/bge-small-en-v1.5` | 384-dim English model |
| `similarity_threshold` | `0.35` | Min cosine similarity for vector hits |
| `batch_size` | `32` | Chunks embedded per batch on process |

Documents uploaded **before** migration 003 must be **re-processed** to populate embeddings.

Ask responses include `retrieval_method` (`vector` or `keyword`) — vector search is tried first; keyword matching is the fallback.

---

Supabase may sign access tokens with **ES256** (new signing keys) or **HS256** (legacy JWT secret). The backend picks the method from the token header `alg`:

| Token `alg` | Required in `backend/.env` |
|-------------|----------------------------|
| `ES256` / `RS256` | `SUPABASE_URL` (must match your frontend project URL) |
| `HS256` | `SUPABASE_JWT_SECRET` from Supabase **Settings → API → JWT Secret** |

After updating `.env`, restart uvicorn and refresh the dashboard. With `APP_DEBUG=true`, backend logs include the underlying JWT error.

Check your token at [jwt.io](https://jwt.io) — paste the `access_token` from the browser session.

---

## API endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/health` | No | Liveness + correlation ID |
| GET | `/ready` | No | Redis, Neo4j, Supabase readiness |
| GET | `/me` | **Bearer JWT** | Current user profile (Step 1.5) |
| POST | `/upload` | **Bearer JWT** | Upload Excel or document (Step 1.6) |
| GET | `/upload/config` | No | Allowed extensions and max size from `config.yaml` |
| GET | `/documents` | **Bearer JWT** | List your uploaded files |
| GET | `/documents/{id}` | **Bearer JWT** | Document metadata + status |
| POST | `/documents/{id}/process` | **Bearer JWT** | Parse, chunk, summarize (Step 1.7) |
| GET | `/documents/{id}/summary` | **Bearer JWT** | Cached summary from Redis/Postgres |
| POST | `/documents/{id}/ask` | **Bearer JWT** | RAG chat over document chunks |
| GET | `/docs` | No | OpenAPI UI |

### Calling `/me` from curl

```bash
# Replace TOKEN with Supabase access_token from browser session
curl http://localhost:8000/me -H "Authorization: Bearer TOKEN"
```

---

## Project layout

```text
backend/
├── app/
│   ├── main.py              # FastAPI + pycorekit middleware
│   ├── core/
│   │   ├── auth.py          # JWT decode
│   │   ├── deps.py          # get_current_user
│   │   ├── config.py
│   │   ├── exceptions.py
│   │   ├── redis_client.py
│   │   └── neo4j_client.py
│   ├── api/routes/
│   │   ├── health.py
│   │   └── me.py
│   └── services/
│       ├── document_service.py
│       ├── embeddings.py
│       ├── document_text.py
│       ├── llm_client.py
│       └── upload.py
├── logs/                    # gitignored — pycorekit logs
├── environment.yml
└── requirements.txt
```

See [docs/IMPLEMENTATION.md](../docs/IMPLEMENTATION.md) for feature progress.

---

## After dependency changes

```powershell
conda activate insightlab
pip install -r requirements.txt
```

Or recreate the env:

```powershell
conda env remove -n insightlab
conda env create -f environment.yml
pip install -e ..\the-learning-curve-labs\pycorekit
```
