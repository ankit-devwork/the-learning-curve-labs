# Deployment options (capstone / demo)

## Recommended paths

| Goal | Best option | Guide |
|------|-------------|-------|
| Develop and demo on your laptop | **Local Docker** | [DOCKER.md](DOCKER.md) |
| Public cloud URL with persistence | **AWS EC2 + ECR** | [EC2.md](EC2.md) |
| Temporary public link (no cloud VM) | **Local Docker + tunnel** | Below |

---

## Local Docker (free, full features)

```powershell
cd D:\Mine\Learining\GenAI\python\the-learning-curve-labs
copy genai-doc-assistant-capstone\.env.example genai-doc-assistant-capstone\.env
# Set GROQ_API_KEY in .env

docker compose -f genai-doc-assistant-capstone/docker-compose.yml up --build
```

Open http://localhost:8501 — demo via screen share, recording, or slides.

| Pros | Cons |
|------|------|
| Full app works | Not a public URL unless you tunnel |
| Uploads + vector DB persist in Docker volumes | Requires Docker on your machine |

---

## AWS EC2 + ECR (cloud, persistent)

Push two image **tags** into one ECR repo (`digital-worker-studio`), run `docker-compose.ecr.yml` on EC2.

```bash
ECR_REGISTRY=123456789012.dkr.ecr.us-east-1.amazonaws.com \
ECR_REPOSITORY=digital-worker-studio \
./genai-doc-assistant-capstone/scripts/push-ecr.sh
```

Full verified steps (disk, security group, image naming): **[EC2.md](EC2.md)**

| Pros | Cons |
|------|------|
| Public URL, real disk, enough RAM for embeddings | ~$30–60/mo for t3.medium/large |
| Same Docker stack as local | You manage the VM |

---

## Free public URL: local app + tunnel

Expose your **local** Streamlit app with a temporary public link. No EC2 payment.

### Option A — Cloudflare Tunnel (free)

1. Install [cloudflared](https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation/)
2. Start the app locally (Docker command above)
3. In another terminal:

```powershell
cloudflared tunnel --url http://localhost:8501
```

4. Share the `*.trycloudflare.com` URL.

### Option B — ngrok (free tier)

```powershell
ngrok http 8501
```

> Tunnels expose **Streamlit only**. The backend stays on the Docker network (`http://backend:8000`) — Streamlit reaches it internally.

---

## Streamlit Community Cloud (UI only)

[share.streamlit.io](https://share.streamlit.io) hosts the UI for free, but you still need a **public backend API** (EC2, or a tunneled local backend).

**Secrets:**

```toml
BACKEND_URL = "https://your-api-url"
```

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

| Your goal | Best choice |
|-----------|-------------|
| Capstone on your laptop | **Local Docker** |
| Instructor opens a link remotely | **Local Docker + Cloudflare Tunnel** |
| Always-on public demo with persistence | **EC2 + ECR** ([EC2.md](EC2.md)) |
