# Deploy to Render

This guide deploys the **FastAPI backend** and **Streamlit UI** as two Render web services using the Blueprint at `genai-doc-assistant-capstone/render.yaml`.

## Architecture on Render

```text
genai-doc-assistant-capstone/render.yaml
├── genai-backend (Docker, Free)
│   ├── FastAPI + ChromaDB + embeddings
│   ├── Ephemeral storage (data lost on redeploy/restart)
│   └── Health check: /health
│
└── genai-streamlit (Docker, Free)
    ├── Streamlit UI
    ├── BACKEND_URL → backend public URL (RENDER_EXTERNAL_URL)
    └── Health check: /_stcore/health
```

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| [Render account](https://render.com) | Free tier (no card required for free services) |
| GitHub repo connected | `ankit-devwork/the-learning-curve-labs` |
| Groq API key | [console.groq.com](https://console.groq.com) |

> **Free tier:** No persistent disk — re-upload documents after each redeploy or long idle spin-down. Fine for capstone demos.

> **Build time:** First Docker build downloads PyTorch + embedding models and can take **15–25 minutes**.

## Option A — Blueprint (recommended)

1. Push this repo to GitHub (branch `main`).
2. In Render: **New → Blueprint**.
3. Connect the `the-learning-curve-labs` repository.
4. Set the Blueprint file path to **`genai-doc-assistant-capstone/render.yaml`** (not the repo root — this monorepo has multiple projects).
5. Render shows two services from the Blueprint.
6. When prompted, set:
   - `GROQ_API_KEY` — your Groq API key
   - `CORS_ALLOW_ORIGINS` — your Streamlit URL, e.g. `https://genai-streamlit.onrender.com`
7. Click **Apply** and wait for both services to deploy.
8. Open the **genai-streamlit** service URL in your browser.

### Why is `dockerContext` the repo root?

The backend Docker image copies **both** `genai-doc-assistant-capstone/` and `pycorekit/` from the monorepo. Paths in `render.yaml` are relative to the **repository root**, even though the Blueprint file lives inside the capstone folder.

Do **not** set `rootDir` on Docker services here — it breaks `dockerfilePath` resolution for this monorepo layout.

## Option B — Manual setup

### 1. Backend service

| Setting | Value |
|---------|-------|
| Type | Web Service |
| Runtime | Docker |
| Root Directory | *(leave empty — repo root)* |
| Dockerfile | `genai-doc-assistant-capstone/Dockerfile` |
| Docker Context | `.` |
| Plan | Free |
| Health Check Path | `/health` |

**Environment variables:**

```bash
GROQ_API_KEY=gsk_your_key
APP_ENV=production
APP_RAG__SEMANTIC_DEDUPE=false
CORS_ALLOW_ORIGINS=https://your-streamlit.onrender.com
```

No persistent disk — skip the Disks section in Render (data is ephemeral).

### 2. Streamlit service

| Setting | Value |
|---------|-------|
| Type | Web Service |
| Runtime | Docker |
| Dockerfile | `genai-doc-assistant-capstone/front-end/streamlit/Dockerfile` |
| Docker Context | `.` |
| Plan | Free |
| Health Check Path | `/_stcore/health` |

**Environment variables:**

```bash
BACKEND_URL=https://genai-backend-qspw.onrender.com
```

Use your backend’s public URL from the Render dashboard (copy from **genai-backend** service page).

## Verify deployment

```bash
# Backend liveness
curl https://genai-backend.onrender.com/health

# Backend readiness (loads embedding model — may take 30–60s first time)
curl https://genai-backend.onrender.com/ready

# API docs
https://genai-backend.onrender.com/docs
```

In the Streamlit UI: upload a PDF, then ask a question.

## Environment variable reference

| Variable | Service | Required | Description |
|----------|---------|----------|-------------|
| `GROQ_API_KEY` | backend | Yes | LLM API key |
| `CORS_ALLOW_ORIGINS` | backend | Recommended | Streamlit public URL(s), comma-separated |
| `BACKEND_URL` | streamlit | Auto in Blueprint | Backend public URL (`RENDER_EXTERNAL_URL`) |
| `BACKEND_HOSTPORT` | streamlit | Docker Compose only | Private `host:port` (local network) |
| `APP_RAG__SEMANTIC_DEDUPE` | backend | No | Set `false` on Render for faster uploads |
| `PORT` | both | Auto | Injected by Render; do not override |

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Build timeout | Retry deploy; first build is slow due to PyTorch. |
| 502 on first request | Service cold-starting; wait 30–60s. Embedding model loads on first query. |
| Upload works but docs gone after redeploy | Expected on free tier — re-upload documents. |
| Streamlit cannot reach API | Set `BACKEND_URL=https://your-backend.onrender.com` on Streamlit (remove `BACKEND_HOSTPORT` if set). |
| CORS errors | Set `CORS_ALLOW_ORIGINS` to your Streamlit URL on the backend. |
| Out of memory | Upgrade backend to Standard (2 GB RAM). Embedding model needs ~1 GB+. |

## Cost

| Service | Plan | Cost |
|---------|------|------|
| genai-backend | Free | $0 |
| genai-streamlit | Free | $0 |

**Trade-offs:** 512 MB RAM per service, spin-down after ~15 min idle, no data persistence.

**OOM warning:** The backend may exceed 512 MB when loading embeddings. `render.yaml` uses the smaller `all-MiniLM-L6-v2` model. If deploy logs show `Out of memory (used over 512Mi)`, use [FREE_DEPLOY.md](FREE_DEPLOY.md) (local Docker) or upgrade the backend to **Standard** (2 GB).

For a more reliable demo without cloud limits, see **[FREE_DEPLOY.md](FREE_DEPLOY.md)** (local Docker + optional tunnel).
