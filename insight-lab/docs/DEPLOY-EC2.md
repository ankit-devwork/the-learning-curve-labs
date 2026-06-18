# Deploy InsightLab on EC2 (pilot)

Minimal production setup: **FastAPI on EC2** (nginx + uvicorn + systemd), **Next.js on Vercel**, **Supabase** hosted.

## Before you deploy

1. **Merge PR #12** if still open (upload timeout + LangSmith noise fixes help on EC2).
2. **Run Supabase migrations** `001` through `007` on your project (SQL Editor or CLI). Migration **007** is required for multi-doc search, topic progress, and adaptive quizzes.
3. Choose domains, for example:
   - Frontend: `https://app.yourdomain.com` (Vercel)
   - API: `https://api.yourdomain.com` (EC2 + nginx)

## Architecture

```text
User → Vercel (Next.js) → Supabase Auth + DB + Storage
                       ↘ EC2 nginx → uvicorn → Groq / Redis / Neo4j
```

| Component | Where |
|-----------|-------|
| Frontend | Vercel (recommended) or same EC2 |
| API | EC2 (this guide) |
| Auth, Postgres, Storage, pgvector | Supabase (hosted) |
| Cache & rate limits | Redis on EC2 or [Upstash](https://upstash.com) |
| Knowledge graph | Neo4j via `docker compose` on EC2 |
| LLM | Groq (via LiteLLM) |

---

## 1. EC2 instance

| Setting | Recommendation |
|---------|----------------|
| AMI | Ubuntu 22.04 LTS |
| Size | **t3.small** minimum; **t3.medium** safer for PDF + embeddings |
| Disk | 20 GB+ (FastEmbed model cache ~500 MB) |
| Security group | **22** (your IP only), **80**, **443** — do **not** expose port 8000 publicly |

---

## 2. Base setup

SSH into the instance:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y git nginx certbot python3-certbot-nginx docker.io docker-compose-plugin

sudo usermod -aG docker $USER
# Log out and back in so the docker group applies
```

---

## 3. Clone and install backend

```bash
git clone https://github.com/ankit-devwork/the-learning-curve-labs.git
cd the-learning-curve-labs/insight-lab

# Redis + Neo4j (adaptive quiz and graph sync use Neo4j)
docker compose up -d

cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip

# pycorekit (required)
pip install -e ../../pycorekit
# Or from GitHub:
# pip install "git+https://github.com/ankit-devwork/the-learning-curve-labs.git@main#subdirectory=pycorekit"

pip install -r requirements.txt
```

---

## 4. Backend environment

Copy `backend/.env.example` to `backend/.env` and set:

```bash
# Production — hides /docs and limits /ready detail
APP_APP__ENV=production
APP_DEBUG=false

# Supabase (Project Settings → API)
SUPABASE_URL=https://xxxx.supabase.co
SUPABASE_SERVICE_ROLE_KEY=eyJ...
SUPABASE_JWT_SECRET=your-jwt-secret

# Neo4j (docker-compose on same host)
NEO4J_URI=bolt://127.0.0.1:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=insightlab_dev_password   # match NEO4J_AUTH in docker-compose.yml

# Redis — pick ONE approach:

# A) Local Redis from docker-compose (simplest on EC2)
REDIS_HOST=127.0.0.1
REDIS_PORT=6379

# B) Upstash (no Redis container needed)
# UPSTASH_REDIS_REST_URL=https://...
# UPSTASH_REDIS_REST_TOKEN=...

# LLM
GROQ_API_KEY=gsk_...

# CORS — frontend origin(s), comma-separated, no trailing slash
CORS_ALLOW_ORIGINS=https://app.yourdomain.com

LOG_DIR=/var/log/insightlab
```

**Notes:**

- `APP_APP__ENV=production` sets `app.env` in `config.yaml` (hides OpenAPI docs in production).
- `APP_ENV` in `.env` is the Pydantic `Settings` field — separate from yaml; set both for clarity.
- If `/ready` returns **503 degraded**, check Redis, Neo4j, and Supabase keys.

See [backend/.env.example](../backend/.env.example) for the full variable list.

---

## 5. systemd service (uvicorn)

```bash
sudo mkdir -p /var/log/insightlab
sudo chown $USER:$USER /var/log/insightlab
```

Create `/etc/systemd/system/insightlab-api.service`:

```ini
[Unit]
Description=InsightLab FastAPI
After=network.target docker.service
Requires=docker.service

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/the-learning-curve-labs/insight-lab/backend
Environment="PATH=/home/ubuntu/the-learning-curve-labs/insight-lab/backend/.venv/bin"
ExecStart=/home/ubuntu/the-learning-curve-labs/insight-lab/backend/.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 2
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Adjust `User`, paths, and worker count for your setup. Use **1 worker** on t3.small if memory is tight (FastEmbed loads per worker).

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now insightlab-api
sudo systemctl status insightlab-api

curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/ready
```

---

## 6. nginx reverse proxy

Create `/etc/nginx/sites-available/insightlab-api`:

```nginx
server {
    listen 80;
    server_name api.yourdomain.com;

    client_max_body_size 12M;   # 10 MB upload limit + headroom

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Document process / quiz / multi-doc can take >60s
        proxy_connect_timeout 300s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }
}
```

Enable and add TLS:

```bash
sudo ln -s /etc/nginx/sites-available/insightlab-api /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
sudo certbot --nginx -d api.yourdomain.com
```

---

## 7. Frontend (recommended: Vercel)

In Vercel project env vars (or `frontend/.env.production`):

```bash
NEXT_PUBLIC_SUPABASE_URL=https://xxxx.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=eyJ...
NEXT_PUBLIC_API_URL=https://api.yourdomain.com
NEXT_PUBLIC_SHOW_DEV_PANEL=false
```

Deploy from the `frontend/` directory. Point `app.yourdomain.com` at Vercel.

**Same EC2 option:** run `npm run build && npm start -p 3000` and add a second nginx `server` block proxying to `:3000`. Vercel is less ops for a pilot.

See [frontend/README.md](../frontend/README.md) for Google OAuth setup.

---

## 8. Supabase Auth settings

In Supabase → **Authentication → URL configuration**:

| Field | Value |
|-------|-------|
| Site URL | `https://app.yourdomain.com` |
| Redirect URLs | `https://app.yourdomain.com/**`, `http://localhost:3000/**` (keep for local dev) |

If using Google OAuth, add the production redirect URI in Google Cloud Console as well.

---

## 9. Smoke test checklist

After deploy, logged in as a test user:

| Step | Verify |
|------|--------|
| Login | Email or Google → lands on `/dashboard` |
| Upload | PDF or Excel → appears in **Your files** |
| Process | Document opens → **Summary** loads |
| Ask | Single-doc question → answer + **Based on** filename |
| Quiz | Generate → submit → score |
| Progress | **Your progress by topic** on dashboard |
| Multi-doc | Select 2+ docs → **Ask** → document picker → answer |

Watch logs during first upload/process (first embedding run downloads the FastEmbed model — expect a slow first request):

```bash
journalctl -u insightlab-api -f
tail -f /var/log/insightlab/*.log
```

---

## 10. Optional: Upstash instead of local Redis

Skip the Redis container if you use Upstash:

```bash
UPSTASH_REDIS_REST_URL=...
UPSTASH_REDIS_REST_TOKEN=...
# Leave REDIS_HOST empty
```

Neo4j still needs to run for graph sync and adaptive quiz unless you defer those features.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| 401 on all API calls | Wrong `SUPABASE_JWT_SECRET`; or empty `SUPABASE_URL` in shell env overriding `.env` |
| CORS error in browser | Add exact frontend URL to `CORS_ALLOW_ORIGINS` (no trailing slash) |
| 502 / timeout on upload or process | nginx `proxy_read_timeout 300s`; merge PR #12 |
| `/ready` returns 503 | Redis or Neo4j down — run `docker compose ps` |
| Mastery / multi-doc broken | Migration **007** not applied on Supabase |
| `fastembed` / import errors | `pip install fastembed` in the venv; restart systemd service |

---

## Related docs

- [ARCHITECTURE.md](ARCHITECTURE.md) — system design
- [IMPLEMENTATION.md](IMPLEMENTATION.md) — feature checklist
- [backend/README.md](../backend/README.md) — local backend setup and API routes
