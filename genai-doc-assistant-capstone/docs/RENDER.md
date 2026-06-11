# Deploy to Render

This guide deploys the **FastAPI backend** and **Streamlit UI** as two Render web services using the Blueprint at `genai-doc-assistant-capstone/render.yaml`.

## Architecture on Render

```text
genai-doc-assistant-capstone/render.yaml
├── genai-backend (Docker, Starter + 1GB disk)
│   ├── FastAPI + ChromaDB + embeddings
│   ├── Persistent disk at /app/data (uploads + vector store)
│   └── Health check: /health
│
└── genai-streamlit (Docker, Starter)
    ├── Streamlit UI
    ├── BACKEND_HOSTPORT → private network to backend
    └── Health check: /_stcore/health
```

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| [Render account](https://render.com) | Starter plan recommended (~$7/service/month) |
| GitHub repo connected | `ankit-devwork/the-learning-curve-labs` |
| Groq API key | [console.groq.com](https://console.groq.com) |

> **Why Starter?** The backend needs a **persistent disk** so uploaded documents and the Chroma vector DB survive redeploys. Disks require a paid plan.

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

## Option B — Manual setup

### 1. Backend service

| Setting | Value |
|---------|-------|
| Type | Web Service |
| Runtime | Docker |
| Root Directory | *(leave empty — repo root)* |
| Dockerfile | `genai-doc-assistant-capstone/Dockerfile` |
| Docker Context | `.` |
| Plan | Starter |
| Health Check Path | `/health` |

**Environment variables:**

```bash
GROQ_API_KEY=gsk_your_key
APP_ENV=production
APP_RAG__SEMANTIC_DEDUPE=false
CORS_ALLOW_ORIGINS=https://your-streamlit.onrender.com
```

**Persistent disk:**

| Setting | Value |
|---------|-------|
| Mount path | `/app/data` |
| Size | 1 GB |

### 2. Streamlit service

| Setting | Value |
|---------|-------|
| Type | Web Service |
| Runtime | Docker |
| Dockerfile | `genai-doc-assistant-capstone/front-end/streamlit/Dockerfile` |
| Docker Context | `.` |
| Plan | Starter |
| Health Check Path | `/_stcore/health` |

**Environment variables:**

```bash
BACKEND_HOSTPORT=genai-backend:10000
```

Use the backend’s private **host:port** from the Render dashboard (Networking → Private). Or set a public URL:

```bash
BACKEND_URL=https://genai-backend.onrender.com
```

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
| `BACKEND_HOSTPORT` | streamlit | Auto in Blueprint | Private `host:port` from Render |
| `BACKEND_URL` | streamlit | Alternative | Public backend URL |
| `APP_RAG__SEMANTIC_DEDUPE` | backend | No | Set `false` on Render for faster uploads |
| `PORT` | both | Auto | Injected by Render; do not override |

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Build timeout | Use Starter plan; rebuild. First build is slow due to PyTorch. |
| 502 on first request | Service cold-starting; wait 30–60s. Embedding model loads on first query. |
| Upload works but docs gone after redeploy | Attach persistent disk at `/app/data` on backend. |
| Streamlit cannot reach API | Check `BACKEND_HOSTPORT` or `BACKEND_URL` in Streamlit env vars. |
| CORS errors | Set `CORS_ALLOW_ORIGINS` to your Streamlit URL on the backend. |
| Out of memory | Upgrade backend to Standard (2 GB RAM). Embedding model needs ~1 GB+. |

## Cost estimate

| Service | Plan | Approx. cost |
|---------|------|--------------|
| genai-backend | Starter + 1GB disk | ~$9/month |
| genai-streamlit | Starter | ~$7/month |

Free tier is not recommended: limited RAM, no persistent disk, and services spin down after inactivity.
