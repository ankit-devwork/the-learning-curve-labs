# Free deployment options (capstone / demo)

Render **Starter + disk** costs ~$16/month for two services. For a capstone demo, use one of these **free** options instead.

## Recommended: run locally (100% free, full features)

Already set up in this repo. No cloud payment, no RAM limits, data persists in Docker volumes.

```powershell
cd D:\Mine\Learining\GenAI\python\the-learning-curve-labs
copy genai-doc-assistant-capstone\.env.example genai-doc-assistant-capstone\.env
# Set GROQ_API_KEY in .env

docker compose -f genai-doc-assistant-capstone/docker-compose.yml up --build
```

Open http://localhost:8501 — demo via screen share, recorded video, or slides.

| Pros | Cons |
|------|------|
| Full app works | Not a public URL unless you tunnel (below) |
| Uploads + vector DB persist | Requires Docker on your machine |

---

## Free public URL: local app + tunnel (best free “hosted” demo)

Expose your **local** Streamlit app with a temporary public link. No Render payment.

### Option A — Cloudflare Tunnel (free, no card)

1. Install [cloudflared](https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation/)
2. Start the app locally (Docker command above)
3. In another terminal:

```powershell
cloudflared tunnel --url http://localhost:8501
```

4. Share the `*.trycloudflare.com` URL with your instructor.

### Option B — ngrok (free tier)

```powershell
ngrok http 8501
```

Share the forwarded URL. Free tier has session limits.

> Tunnels expose **Streamlit only**. The backend stays on `localhost:8000` inside Docker — that’s fine because Streamlit talks to it on the Docker network.

---

## Render free tier (default `render.yaml`)

The main Blueprint already uses **free** plans with **no disk**:

1. Blueprint path: **`genai-doc-assistant-capstone/render.yaml`**
2. Accept limitations:

| Limitation | Impact |
|------------|--------|
| **512 MB RAM** | May crash when loading embedding model; first query often fails |
| **No persistent disk** | Uploads + vector DB **lost** on every redeploy/restart |
| **Spin down** | Cold start ~30–60s after 15 min idle |
| **Slow builds** | First deploy still 15–25 min |

Use only for a **short live demo** after a fresh upload — not for production.

### Out of memory on Render (`used over 512Mi`)

The backend loads **PyTorch + embedding models**. Render **free tier = 512 MB RAM**, which often is not enough for the default `all-mpnet-base-v2` model.

**Symptoms:** deploy logs show `Out of memory (used over 512Mi)`, `/documents` returns HTML errors, Streamlit shows `Invalid JSON response`.

**Options (pick one):**

| Option | Cost | Reliability |
|--------|------|-------------|
| **Local Docker** | Free | Best for capstone |
| **Render + smaller model** | Free | `render.yaml` sets `all-MiniLM-L6-v2` — may still OOM |
| **Render Standard backend** | ~$25/mo (2 GB RAM) | Works with full model |

**Recommended for capstone:** local Docker + screen recording or Cloudflare Tunnel — no Render backend needed.

---

## Streamlit Community Cloud (free, UI only)

[share.streamlit.io](https://share.streamlit.io) — free Streamlit hosting from GitHub.

**Catch:** You still need a **public backend API**. Options:

- Tunnel only the backend: `cloudflared tunnel --url http://localhost:8000` → set `BACKEND_URL` in Streamlit Cloud secrets
- Or deploy backend to a free tier elsewhere (HF Spaces, Fly.io)

**Streamlit Cloud settings:**

| Setting | Value |
|---------|-------|
| Repository | `the-learning-curve-labs` |
| Branch | `main` |
| Main file | `genai-doc-assistant-capstone/front-end/streamlit/chat.py` |
| App URL path | (default) |

**Secrets** (in Streamlit Cloud → Settings → Secrets):

```toml
BACKEND_URL = "https://your-tunnel-or-api-url"
```

The capstone Streamlit app alone does **not** include the backend — you must host API separately or use local Docker + tunnel.

---

## Hugging Face Spaces (free, Docker)

[huggingface.co/spaces](https://huggingface.co/new-space) — free CPU, good for ML demos.

- Create a **Docker** Space
- Push a combined image or use their Docker SDK
- More setup than local Docker; better if you want a permanent public link without your PC running

See HF docs: [Docker Spaces](https://huggingface.co/docs/hub/spaces-sdks-docker)

---

## What stays free regardless

| Service | Cost |
|---------|------|
| **Groq API** | Free tier with rate limits — [console.groq.com](https://console.groq.com) |
| **GitHub** | Free for public/private repos |
| **Local Docker** | Free |
| **Langfuse / LangSmith** | Optional; skip for capstone |

---

## Quick decision

| Your goal | Best free choice |
|-----------|------------------|
| Capstone presentation on your laptop | **Local Docker** |
| Instructor opens a link remotely | **Local Docker + Cloudflare Tunnel** |
| Must be “in the cloud” with zero payment | **Render free** (`render.yaml`) — expect fragility |
| Long-term free public demo | **HF Spaces** or **Streamlit Cloud + tunneled API** |

For most capstone submissions, **local Docker + screenshots/video** or **local + Cloudflare Tunnel** is enough and avoids payment entirely.
