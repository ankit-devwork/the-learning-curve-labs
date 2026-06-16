# InsightLab

**Make sense of your data and documents** — Excel charts, document chat, and AI-generated quizzes in one platform.

Part of [The Learning Curve Labs](https://github.com/ankit-devwork/the-learning-curve-labs).

## Features (planned)

| Mode | Input | Output |
|------|-------|--------|
| **Excel insights** | `.xlsx`, `.csv` | Auto charts, narrative insights, optional data chat |
| **Document intelligence** | `.pdf`, `.txt`, `.docx` | Summary, RAG chat with citations |
| **Quiz generator** | Any ingested document | Single-choice / MCQ quizzes with scoring |

## Stack

| Layer | Technology |
|-------|------------|
| Frontend | Next.js, shadcn/ui, Tailwind |
| Auth & data | Supabase (Auth, Postgres, Storage, pgvector) |
| Knowledge graph | Neo4j |
| Cache & resilience | Redis / Upstash (cache, rate limit, retry) |
| Backend | FastAPI, LangGraph, LiteLLM |
| Observability | pycorekit, Langfuse |

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full system design.

## Repository structure

```text
insight-lab/
├── frontend/          # Next.js app (auth, dashboard, UI)
├── backend/           # FastAPI + LangGraph agents
├── supabase/          # SQL migrations and RLS policies
├── docs/              # Architecture and runbooks
└── docker-compose.yml # Local Neo4j + Redis
```

## Quick start (local)

### 1. Clone and configure

```bash
git clone https://github.com/ankit-devwork/insight-lab.git
cd insight-lab
cp .env.example .env
cp backend/.env.example backend/.env
# Fill in Supabase, Neo4j, Redis/Upstash, and LLM keys
```

### 2. Start infrastructure

```bash
docker compose up -d
```

Starts local **Redis** and **Neo4j**. Use a hosted [Supabase](https://supabase.com) project for auth and Postgres.

### 3. Backend (Conda)

```powershell
cd backend
conda env create -f environment.yml
conda activate insightlab
copy .env.example .env
uvicorn app.main:app --reload --port 8000
```

See [backend/README.md](backend/README.md) for Conda updates and troubleshooting.

- API: http://localhost:8000
- Docs: http://localhost:8000/docs

### 4. Frontend (Next.js + Supabase Auth)

```powershell
cd frontend
copy .env.local.example .env.local
# Set NEXT_PUBLIC_SUPABASE_URL and NEXT_PUBLIC_SUPABASE_ANON_KEY

npm install
npm run dev
```

- App: http://localhost:3000
- Login: http://localhost:3000/login

See [frontend/README.md](frontend/README.md) for Google OAuth setup in Supabase.

## Environment variables

| Variable | Where | Purpose |
|----------|-------|---------|
| `NEXT_PUBLIC_SUPABASE_URL` | frontend | Supabase project URL |
| `NEXT_PUBLIC_SUPABASE_ANON_KEY` | frontend | Supabase anon key |
| `SUPABASE_URL` | backend | Supabase URL |
| `SUPABASE_SERVICE_ROLE_KEY` | backend | Server-side Supabase access |
| `SUPABASE_JWT_SECRET` | backend | Verify user JWTs |
| `NEO4J_URI` | backend | Neo4j bolt URI |
| `NEO4J_USER` / `NEO4J_PASSWORD` | backend | Neo4j credentials |
| `REDIS_HOST` or Upstash vars | backend | Cache & rate limiting |
| `GROQ_API_KEY` | backend | LLM provider (via LiteLLM) |

See `.env.example` and `backend/.env.example` for the full list.

## Development status

See [docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md) for the full checklist.

- [x] Architecture documented
- [x] Repo structure and MIT license
- [x] Backend health + `/ready` checks
- [x] Supabase schema + Storage
- [x] Next.js auth (email + Google) + dashboard
- [x] **pycorekit + JWT auth + `GET /me`**
- [ ] File upload API
- [ ] Excel, document, and quiz pipelines

## Related projects

- [genai-doc-assistant-capstone](https://github.com/ankit-devwork/the-learning-curve-labs/tree/main/genai-doc-assistant-capstone) — multi-agent RAG reference
- [pycorekit](https://github.com/ankit-devwork/the-learning-curve-labs/tree/main/pycorekit) — shared logging, cache, tracing (**required by backend**)

## License

Licensed under the [MIT License](LICENSE).

The **InsightLab** name and branding are not covered by the MIT license.
