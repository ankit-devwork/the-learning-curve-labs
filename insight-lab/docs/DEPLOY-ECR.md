# Deploy InsightLab with ECR + EC2

**Recommended production pilot:** build the API image on your **Windows** machine, push to **Amazon ECR**, pull and run on **EC2** with Docker Compose. Frontend on **Vercel**; Supabase hosted.

**Repository:** [github.com/ankit-devwork/insight-lab](https://github.com/ankit-devwork/insight-lab)

For the manual git-clone + Python venv path, see [DEPLOY-EC2.md](DEPLOY-EC2.md).

---

## Architecture

```text
Windows (local)                         AWS
───────────────                         ───
docker build  ──►  ECR                  EC2
docker push         insight-lab:api-latest   ├── docker compose -f docker-compose.ecr.yml
                                             │     ├── api (ECR image) → 127.0.0.1:8000
                                             │     ├── redis
                                             │     └── neo4j
                                             └── nginx :443 → localhost:8000

Vercel (Next.js) ──► Supabase + EC2 API
```

| Component | Where |
|-----------|-------|
| Frontend | Vercel |
| API | **Docker image from ECR** — no Python/venv on EC2 |
| Redis + Neo4j | Same EC2 (`docker-compose.ecr.yml`) |
| Auth, Postgres, Storage | Supabase (hosted) |
| TLS | nginx + Certbot on EC2 |

---

## What you do **not** need on EC2

| Skip this | Why |
|-----------|-----|
| `python3 -m venv` | API runs inside the Docker image |
| `pip install -r requirements.txt` | Already baked into the ECR image |
| `uvicorn` / systemd for API | Container runs uvicorn |
| `docker compose up -d` (default file) | Only starts Redis + Neo4j — **not** the API |

---

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| AWS account | ECR + EC2 |
| AWS CLI | `aws configure` on Windows |
| Docker Desktop | Windows — build and push |
| EC2 IAM role | **AmazonEC2ContainerRegistryReadOnly** (pull from ECR) |
| Supabase | Migrations **001–007** applied |
| Groq API key | LLM provider |
| Security group | Inbound: **22** (My IP), **80**, **443**. **Do not** open **8000** publicly |

### Recommended EC2 sizing

| Resource | Recommendation |
|----------|----------------|
| Instance | **t3.medium** (4 GB RAM) |
| AMI | Ubuntu 22.04 or 24.04 LTS |
| EBS | **20–30 GB** (Docker images + Neo4j + logs) |

---

## Part 0 — Connect to EC2 from Windows

```powershell
ssh -i "D:\path\to\your-key.pem" ubuntu@YOUR_EC2_PUBLIC_IP
```

| Item | Value |
|------|-------|
| Username | `ubuntu` (Ubuntu AMI) |
| Port | 22 (security group: **My IP** only) |

**Connection timed out?** Update the SSH rule — your home IP may have changed.

**Permission denied?** Wrong `.pem` file or wrong username.

Optional `~/.ssh/config` entry:

```text
Host insightlab-ec2
    HostName YOUR_EC2_PUBLIC_IP
    User ubuntu
    IdentityFile D:\path\to\your-key.pem
```

Then: `ssh insightlab-ec2`

---

## Part 1 — Build and push (Windows)

```powershell
cd D:\Mine\Learining\GenAI\python\insight-lab
aws configure   # once
.\scripts\push-ecr.ps1
```

Success looks like:

```text
Pushing 321204595484.dkr.ecr.us-east-1.amazonaws.com/insight-lab:api-latest ...
api-latest: digest: sha256:... size: ...
```

Linux/macOS: `./scripts/push-ecr.sh`

---

## Part 2 — EC2 setup (once)

### 1. Security group + IAM

- Inbound: **22** (My IP), **80**, **443**
- Attach IAM role **AmazonEC2ContainerRegistryReadOnly** to the instance

### 2. Install Docker + nginx

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y git nginx certbot python3-certbot-nginx docker.io docker-compose-v2

sudo usermod -aG docker ubuntu
```

Log out and SSH back in. Verify Compose (note the **space**, not a hyphen):

```bash
docker --version
docker compose version
```

**Do not** rely on `docker-compose` (hyphen) — on Ubuntu it is often **not** installed. Use **`docker compose`**.

If `docker compose` is missing:

```bash
sudo apt install -y docker-compose-v2
# or install docker-compose-plugin from Docker's official apt repo (see Troubleshooting)
```

Reboot if apt reports **"System restart required"**:

```bash
sudo reboot
```

### 3. Clone repo (compose files + env only)

```bash
git clone https://github.com/ankit-devwork/insight-lab.git
cd insight-lab

cp backend/.env.example backend/.env
cp .env.ecr.example .env.ecr
nano backend/.env
nano .env.ecr
```

**`.env.ecr` example:**

```bash
ECR_REGISTRY=321204595484.dkr.ecr.us-east-1.amazonaws.com
ECR_REPOSITORY=insight-lab
API_IMAGE_TAG=api-latest
AWS_REGION=us-east-1
NEO4J_AUTH=neo4j/insightlab_dev_password
```

**`backend/.env` essentials:**

```bash
APP_APP__ENV=production
APP_DEBUG=false

SUPABASE_URL=https://xxxx.supabase.co
SUPABASE_SERVICE_ROLE_KEY=...
SUPABASE_JWT_SECRET=...

NEO4J_USER=neo4j
NEO4J_PASSWORD=insightlab_dev_password

GROQ_API_KEY=gsk_...

CORS_ALLOW_ORIGINS=https://your-app.vercel.app
```

Compose sets `REDIS_HOST=redis` and `NEO4J_URI=bolt://neo4j:7687` inside the API container.

### 4. Free port 8000 (if another app uses it)

Check what's bound to 8000:

```bash
docker ps
```

If another stack (e.g. genai capstone) uses **8000**, stop it or InsightLab API will fail to start:

```bash
docker stop genai_backend genai_streamlit   # example names
```

To run **both** apps on one box, change the API port in `docker-compose.ecr.yml` to `127.0.0.1:8002:8000` and point nginx at **8002**.

### 5. Pull and start the **full** stack

Use **`docker-compose.ecr.yml`** — not the default `docker compose up -d`:

```bash
cd ~/insight-lab

aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin 321204595484.dkr.ecr.us-east-1.amazonaws.com

docker compose -f docker-compose.ecr.yml \
  --env-file backend/.env \
  --env-file .env.ecr \
  pull

docker compose -f docker-compose.ecr.yml \
  --env-file backend/.env \
  --env-file .env.ecr \
  up -d
```

### 6. Verify — expect **3** containers

```bash
docker compose -f docker-compose.ecr.yml ps
```

| Container | Status |
|-----------|--------|
| `insightlab-api` | Up (healthy) |
| `insightlab-redis` | Up (healthy) |
| `insightlab-neo4j` | Up (healthy) |

```bash
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/ready
```

If you only see **redis** and **neo4j**, you ran the wrong compose file — go back to step 5.

Logs:

```bash
docker compose -f docker-compose.ecr.yml logs -f api
```

### 7. Expose API for Vercel (no domain — port 8080)

If the frontend is on **Vercel (HTTPS)** and the API has **no domain**, use nginx on **8080** and a **Next.js proxy rewrite** (Part 4). The browser never calls HTTP EC2 directly.

```bash
sudo tee /etc/nginx/sites-available/insightlab-api << 'EOF'
server {
    listen 8080;
    server_name _;

    client_max_body_size 12M;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_connect_timeout 300s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }
}
EOF

sudo ln -sf /etc/nginx/sites-available/insightlab-api /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```

Security group: add inbound **8080** from **Anywhere** (Vercel’s proxy servers need to reach EC2).

Test:

```bash
curl -s http://127.0.0.1:8080/health
curl -s http://YOUR_EC2_PUBLIC_IP:8080/health
```

### 8. nginx + HTTPS (optional — when you have a domain)

Proxy `api.yourdomain.com` → `http://127.0.0.1:8000` with 300s timeouts. Full config: [DEPLOY-EC2.md — nginx](DEPLOY-EC2.md#6-nginx-reverse-proxy).

```bash
sudo certbot --nginx -d api.yourdomain.com
```

Then you can set `NEXT_PUBLIC_API_URL=https://api.yourdomain.com` on Vercel and skip the proxy rewrite.

---

## Part 3 — Deploy updates

**Windows** (after code changes):

```powershell
.\scripts\push-ecr.ps1
```

**EC2:**

```bash
cd ~/insight-lab
docker compose -f docker-compose.ecr.yml --env-file backend/.env --env-file .env.ecr pull
docker compose -f docker-compose.ecr.yml --env-file backend/.env --env-file .env.ecr up -d
```

---

## Part 4 — Frontend on Vercel (HTTP EC2 via proxy)

Browsers block **HTTPS → HTTP** calls. InsightLab proxies API requests through Vercel so the browser only talks to **same-origin HTTPS** (`/api-backend/*`); Vercel forwards to your EC2 nginx on **8080**.

```text
Browser  →  https://your-app.vercel.app/api-backend/upload
         →  Vercel rewrite (server)
         →  http://EC2-IP:8080/upload
         →  nginx → insightlab-api :8000
```

### 1. EC2 nginx on 8080

Complete [step 7](#7-expose-api-for-vercel--no-domain--port-8080) above first.

### 2. Vercel project env vars

In Vercel → Project → Settings → Environment Variables:

| Variable | Value | Notes |
|----------|-------|-------|
| `NEXT_PUBLIC_SUPABASE_URL` | `https://xxxx.supabase.co` | Public |
| `NEXT_PUBLIC_SUPABASE_ANON_KEY` | `eyJ...` | Public |
| `NEXT_PUBLIC_API_URL` | `/api-backend` | Same-origin proxy path |
| `BACKEND_PROXY_URL` | `http://YOUR_EC2_PUBLIC_IP:8080` | **Server-only** — not `NEXT_PUBLIC_` |
| `NEXT_PUBLIC_SHOW_DEV_PANEL` | `false` | |

Redeploy after changing env vars (rewrites read `BACKEND_PROXY_URL` at build time).

### 3. Supabase Auth

| Field | Value |
|-------|-------|
| Site URL | `https://your-app.vercel.app` |
| Redirect URLs | `https://your-app.vercel.app/**`, `http://localhost:3000/**` |

Google OAuth: add Supabase callback URL in Google Cloud Console (unchanged).

### 4. CORS on EC2

With the proxy, the browser does **not** call EC2 directly — **CORS is optional** for Vercel traffic. Keep localhost in CORS for local dev:

```bash
CORS_ALLOW_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
```

### 5. Verify proxy

After Vercel deploy:

```text
https://your-app.vercel.app/api-backend/health
```

Should return `{"status":"ok",...}`.

### 6. Upload size note

Vercel may limit proxied request bodies (~**4.5 MB** on Hobby). InsightLab allows up to **10 MB** uploads. If uploads fail on Vercel but work via `curl` to EC2, use a **HTTPS API** (domain or Cloudflare Tunnel) or host the frontend on EC2.

### Alternative — domain + HTTPS API

If you have a domain, skip the proxy and set:

```bash
NEXT_PUBLIC_API_URL=https://api.yourdomain.com
```

---

## Smoke test

| Step | Verify |
|------|--------|
| `curl http://EC2-IP:8080/health` | API reachable from internet |
| `https://your-app.vercel.app/api-backend/health` | Vercel proxy works |
| Login on Vercel app | Dashboard loads |
| Upload PDF | Summary appears (watch 4.5 MB Vercel limit) |
| Ask / Quiz / Multi-doc | End-to-end works |

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `Unable to locate package docker-compose-plugin` | Use `sudo apt install docker.io docker-compose-v2` instead |
| `docker-compose: command not found` | Use **`docker compose`** (space). Run `docker compose version` |
| Only redis + neo4j running, no API | You ran `docker compose up -d` — use **`-f docker-compose.ecr.yml`** with env files |
| `python3-venv` / venv errors | **Skip venv** — ECR path does not install Python on EC2 |
| Port 8000 already in use | `docker ps` — stop conflicting container or use port **8002** |
| `Cannot pull image` | IAM role **AmazonEC2ContainerRegistryReadOnly**; run ECR `docker login` |
| `push-ecr.ps1` fails on Windows | Repo missing — script creates it; ensure PowerShell fix is pulled (PR #47) |
| API exits on start | `docker compose -f docker-compose.ecr.yml logs api` — check `backend/.env` |
| `/ready` 503 | Redis or Neo4j unhealthy — `docker compose -f docker-compose.ecr.yml ps` |
| CORS errors | With Vercel proxy, browser hits same origin — check proxy URL first |
| Vercel `/api-backend/health` fails | `BACKEND_PROXY_URL` set? EC2 port **8080** open? nginx running? |
| Upload fails on Vercel only | Vercel body limit ~4.5 MB — use HTTPS API or EC2 frontend |
| OOM / slow first request | Use t3.medium; first FastEmbed download is slow |

---

## Command cheat sheet

| Wrong | Correct |
|-------|---------|
| `docker compose up -d` | `docker compose -f docker-compose.ecr.yml --env-file backend/.env --env-file .env.ecr up -d` |
| `docker-compose up -d` | `docker compose ...` (space, not hyphen) |
| `cd backend && python3 -m venv .venv` | Not needed on ECR path |
| Expose port 8000 in security group | Keep 8000 on **127.0.0.1** only; use nginx on **443** |

---

## Related docs

- [DEPLOY-EC2.md](DEPLOY-EC2.md) — manual venv deploy (fallback)
- [ARCHITECTURE.md](ARCHITECTURE.md) — system design
- [backend/README.md](../backend/README.md) — API and env reference
