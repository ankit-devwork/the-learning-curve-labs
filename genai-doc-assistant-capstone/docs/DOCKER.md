# Docker Guide

## Prerequisites

- Docker and Docker Compose v2.24+ (for optional `.env` support)
- Groq API key

## Quick start

```bash
# From repo root (the-learning-curve-labs/)
cp genai-doc-assistant-capstone/.env.example genai-doc-assistant-capstone/.env

# Edit .env and set at minimum:
# GROQ_API_KEY=gsk_your_key_here

docker compose -f genai-doc-assistant-capstone/docker-compose.yml up --build
```

Or from the capstone directory:

```bash
cd genai-doc-assistant-capstone
cp .env.example .env
docker compose up --build
```

| Service | URL |
|---------|-----|
| Streamlit UI | http://localhost:8501 |
| FastAPI | http://localhost:8000 |
| API docs | http://localhost:8000/docs |
| Liveness | http://localhost:8000/health |
| Readiness | http://localhost:8000/ready |

## Architecture

```text
genai-doc-assistant-capstone/docker-compose.yml   (project: genai-doc-assistant-capstone)
├── backend   (monorepo root build context via ..)
│   ├── Copies pycorekit + capstone into image at build time
│   ├── Loads .env from capstone folder (optional)
│   └── Persists data/uploads, data/vector_store, logs in named volumes
│
└── streamlit (monorepo root build context via ..)
    └── BACKEND_URL=http://backend:8000
```

## Production-style compose (default)

```bash
docker compose -f genai-doc-assistant-capstone/docker-compose.yml up --build
```

- `pycorekit` is baked into the backend image at build time
- Uploads, Chroma data, and logs survive container restarts via named volumes
- Healthcheck uses `/health` for fast startup (embedding model loads on first real request)

## Development compose (hot reload)

```bash
docker compose -f genai-doc-assistant-capstone/docker-compose.yml -f genai-doc-assistant-capstone/docker-compose.dev.yml up --build
```

- Mounts capstone + pycorekit source into the backend container
- Enables `uvicorn --reload`
- Mounts Streamlit source for UI edits without rebuild

## Build images manually

```bash
# Backend
docker build -f genai-doc-assistant-capstone/Dockerfile -t genai-backend .

# Streamlit
docker build -f genai-doc-assistant-capstone/front-end/streamlit/Dockerfile -t genai-streamlit .
```

Build context must be the **monorepo root** (`.`), not the capstone folder alone.

## Environment file

| File | Purpose |
|------|---------|
| `.env.example` | Template (committed) |
| `.env` | Your secrets and overrides (not committed) |

Loaded by:
1. Docker Compose `env_file`
2. App `ConfigLoader` at startup (`settings.py`)

## Related docs

- [ARCHITECTURE.md](ARCHITECTURE.md)
- [ARCHITECTURE_DIAGRAM.md](ARCHITECTURE_DIAGRAM.md)
- [CONFIGURATION.md](CONFIGURATION.md)
- [RENDER.md](RENDER.md)

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `exec /app/scripts/start-backend.sh: no such file or directory` (Windows) | Rebuild after pulling latest (`--build`). Scripts are normalized to LF in the image; `.gitattributes` keeps `*.sh` as LF in git. |
| Streamlit cannot reach API | Ensure `BACKEND_URL=http://backend:8000` inside Docker network |
| LLM errors on query | Set `GROQ_API_KEY` in `.env` |
| Slow first query | First request downloads the embedding model — normal |
| Reset vector data | `docker compose -f genai-doc-assistant-capstone/docker-compose.yml down -v` |
