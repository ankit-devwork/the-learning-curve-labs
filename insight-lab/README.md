# InsightLab

**Make sense of your data and documents** — Excel charts, document chat, and AI-generated quizzes in one platform.

Part of [The Learning Curve Labs](https://github.com/ankit-devwork/the-learning-curve-labs).

## Features

| Mode | Input | Output |
|------|-------|--------|
| **Excel insights** | `.xlsx`, `.csv` | Auto charts, custom charts, data chat, narrative insights (notebook workspace) |
| **Document intelligence** | `.pdf`, `.txt`, `.docx` | Summary, RAG chat with citations, multi-document compare |
| **Study notebooks** | Study sets (workspaces) | Upload sources, Studio tools (quiz, flashcards, study guide, audio), course packs, sharing |
| **Quiz & progress** | Any ingested document | Quizzes, scoring, topic progress, practice weak areas |

## Stack

| Layer | Technology |
|-------|------------|
| Frontend | Next.js, shadcn/ui, Tailwind |
| Auth & data | Supabase (Auth, Postgres, Storage, pgvector) |
| Knowledge graph | Neo4j |
| Cache & resilience | Redis / Upstash (cache, rate limit, retry) |
| Backend | FastAPI, LiteLLM (service-layer orchestration) |
| Observability | pycorekit, Langfuse |

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full system design. For production, see [docs/DEPLOY-ECR.md](docs/DEPLOY-ECR.md) (recommended) or [docs/DEPLOY-EC2.md](docs/DEPLOY-EC2.md) (manual venv).

## Repository structure

```text
insight-lab/
├── frontend/          # Next.js app (auth, dashboard, UI)
├── backend/           # FastAPI services + LiteLLM
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

## Deploy (ECR + EC2 + Vercel pilot)

**Recommended (no domain):** API on EC2 via ECR → frontend on Vercel → proxy via `/api-backend`.

See **[docs/DEPLOY-ECR.md](docs/DEPLOY-ECR.md)** for the full step-by-step checklist (EC2, nginx :8080, Vercel env, Supabase Auth).

Fallback (git clone + uvicorn on host): [docs/DEPLOY-EC2.md](docs/DEPLOY-EC2.md).

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
- [x] File upload API (`POST /upload`, dashboard UI)
- [x] Document summary + chat (Step 1.7)
- [x] pgvector embeddings for semantic RAG (Step 1.7b)
- [x] Excel charts pipeline + retry/circuit breaker (Step 1.8)
- [x] Quiz generator (Step 1.9)
- [x] Excel data chat (Phase 2)
- [x] Multi-document chat with document picker (Phase 2)
- [x] Adaptive quizzes + topic progress UI (Phase 2)
- [x] Security hardening (RLS, cache scoping, grounded LLM prompts)

**Ops:** Run all Supabase migrations through `011_phase8_member_rls.sql` for study features, sharing, and member RLS. See [supabase/README.md](supabase/README.md).

## Related projects

- [genai-doc-assistant-capstone](https://github.com/ankit-devwork/the-learning-curve-labs/tree/main/genai-doc-assistant-capstone) — multi-agent RAG reference
- [pycorekit](https://github.com/ankit-devwork/the-learning-curve-labs/tree/main/pycorekit) — shared logging, cache, tracing (**required by backend**)

## License

Licensed under the [MIT License](LICENSE).

The **InsightLab** name and branding are not covered by the MIT license.
