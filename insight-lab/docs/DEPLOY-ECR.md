# Deploy InsightLab with ECR + EC2 + Vercel

**Recommended pilot setup (no custom domain):**

1. Build API image on **Windows** → push to **Amazon ECR**
2. Run API + Redis + Neo4j on **EC2** (Docker)
3. Deploy **frontend only** on **Vercel** (HTTPS)
4. Proxy API calls through Vercel → EC2 over HTTP (port **8080**)

**Repository:** [github.com/ankit-devwork/insight-lab](https://github.com/ankit-devwork/insight-lab)

Manual venv deploy (fallback): [DEPLOY-EC2.md](DEPLOY-EC2.md)

---

## Architecture (no domain)

```text
Windows                          AWS EC2                         Vercel
───────                          ───────                         ──────
push-ecr.ps1 ──► ECR             docker-compose.ecr.yml          Next.js (HTTPS)
                 insight-lab       ├── insightlab-api :8000       └── /api-backend/*
                                   ├── redis                         (proxy rewrite)
                                   ├── neo4j                              │
                                   └── nginx :8080 ◄──────────────────────┘
                                         ▲
                                   Security group :8080

Supabase (hosted) ◄── Auth + DB + Storage
```

| Component | Where |
|-----------|-------|
| Frontend | **Vercel** — root directory `frontend` only |
| API | **ECR → Docker** on EC2 — no Python/venv on host |
| Redis + Neo4j | Docker on same EC2 |
| Auth / Postgres / Storage | Supabase |
| Public API access | nginx **8080** (for Vercel proxy) — **not** port 8000 |

---

## Before you start

| Task | Where |
|------|-------|
| Run Supabase migrations **001–012** | Supabase SQL Editor — see [supabase/README.md](../supabase/README.md) |
| Create Supabase project keys | Settings → API |
| Groq API key | [console.groq.com](https://console.groq.com) |
| AWS CLI configured | `aws configure` on Windows |
| Docker Desktop | Windows (build/push image) |
| EC2 Ubuntu instance | t3.medium recommended, 20–30 GB disk |

### Supabase migrations (001–011)

Run in order in the Supabase SQL Editor (or `supabase db push`):

| # | File | Required for |
|---|------|--------------|
| 001 | `001_initial.sql` | Core schema, workspaces, documents |
| 002 | `002_document_chunks.sql` | Summary + chat chunks |
| 003 | `003_pgvector_embeddings.sql` | Semantic RAG |
| 004 | `004_excel_charts.sql` | Excel analysis |
| 005 | `005_rls_policies.sql` | Table RLS |
| 006 | `006_storage_and_rpc_security.sql` | Storage + RPC lockdown |
| 007 | `007_phase2_graph_mastery_multi_doc.sql` | Graph, mastery, multi-doc |
| 008 | `008_phase3_4_study_features.sql` | Flashcards, study guides |
| 009 | `009_phase6_sharing_quiz_edit.sql` | Sharing, invites, quiz publish |
| 010 | `010_security_hardening.sql` | RLS fixes, editor checks |
| 011 | `011_phase8_member_rls.sql` | Member-aware artifact RLS |
| 012 | `012_profiles_email.sql` | Profile email sync for invites |

Also create a private Storage bucket named **`uploads`**.

---

## Complete walkthrough (checklist)

Use this order — matches a real pilot deploy.

### Phase A — Push API image (Windows)

- [ ] **A1.** `cd insight-lab` → `aws configure`
- [ ] **A2.** `.\scripts\push-ecr.ps1` — creates ECR repo if missing
- [ ] **A3.** Confirm push: `…/insight-lab:api-latest` digest printed

### Phase B — EC2 backend

- [ ] **B1.** Security group inbound: **22** (My IP), **8080** (Anywhere). Optional: **80**, **443** for future domain
- [ ] **B2.** Attach IAM role **AmazonEC2ContainerRegistryReadOnly** to EC2
- [ ] **B3.** SSH: `ssh -i key.pem ubuntu@EC2_IP`
- [ ] **B4.** Install packages (see [Part 2](#part-2--ec2-backend))
- [ ] **B5.** `docker compose version` works (space, not hyphen)
- [ ] **B6.** Clone repo, configure `backend/.env` and `.env.ecr`
- [ ] **B7.** Confirm `docker-compose.ecr.yml` exists (`git pull` if missing)
- [ ] **B8.** Stop conflicting containers on port 8000 (`docker ps`)
- [ ] **B9.** ECR login → `docker compose -f docker-compose.ecr.yml … pull && up -d`
- [ ] **B10.** Verify 3 containers + `curl localhost:8000/health` and `/ready`
- [ ] **B11.** nginx on **8080** → `curl EC2_IP:8080/health` from Windows

### Phase C — Vercel frontend

- [ ] **C1.** Import GitHub repo — **Root Directory = `frontend`** (do **not** deploy FastAPI on Vercel)
- [ ] **C2.** Next.js **≥ 15.5.18** (Vercel blocks vulnerable versions)
- [ ] **C3.** Set Vercel env vars (see [Part 4](#part-4--vercel-frontend))
- [ ] **C4.** Redeploy after env vars
- [ ] **C5.** Test `https://YOUR-APP.vercel.app/api-backend/health`

### Phase D — Supabase Auth

- [ ] **D1.** Site URL = `https://YOUR-APP.vercel.app` (**not** localhost)
- [ ] **D2.** Redirect URLs include `https://YOUR-APP.vercel.app/**`
- [ ] **D3.** Login/signup on Vercel — lands on dashboard, not localhost

### Phase E — Smoke test

- [ ] **E1.** Create or open a study set → upload PDF → opens **document notebook** → Brief tab shows summary
- [ ] **E2.** Ask a question in chat → answer with citations / **Based on** filename
- [ ] **E3.** Studio → Generate quiz → submit → score
- [ ] **E4.** Compare: select 2+ **document** files → multi-doc answer
- [ ] **E5.** Upload Excel → opens **Excel notebook** → Insights, Preview, and Charts tabs after analysis
- [ ] **E6.** (Optional) Share study set → invite link → second user can view sources

---

## What you do **not** need on EC2

| Skip | Why |
|------|-----|
| `python3 -m venv` | API runs in Docker image |
| `pip install` / uvicorn on host | Baked into ECR image |
| `docker compose up -d` (default) | Only starts Redis + Neo4j — **not** the API |
| Opening port **8000** publicly | API stays on `127.0.0.1:8000`; nginx uses **8080** |

---

## Part 1 — Build and push (Windows)

```powershell
cd D:\Mine\Learining\GenAI\python\insight-lab
aws configure
.\scripts\push-ecr.ps1
```

Success:

```text
Login Succeeded
Pushing 321204595484.dkr.ecr.us-east-1.amazonaws.com/insight-lab:api-latest ...
api-latest: digest: sha256:... size: ...
```

**`push-ecr.ps1` fails on “describe-repositories”?** Pull latest script (PR #47) — or create repo manually:

```powershell
aws ecr create-repository --repository-name insight-lab --region us-east-1
```

Linux/macOS: `./scripts/push-ecr.sh`

---

## Part 2 — EC2 backend

### Connect from Windows

```powershell
ssh -i "D:\path\to\your-key.pem" ubuntu@YOUR_EC2_PUBLIC_IP
```

Get public IP on EC2: `curl -s ifconfig.me`

### Security group

| Port | Source | Purpose |
|------|--------|---------|
| **22** | My IP | SSH |
| **8080** | Anywhere | Vercel → API proxy (required without domain) |
| 80, 443 | Anywhere | Optional — for custom domain later |

**Do not** open port **8000** to the internet.

### IAM

EC2 → Instance → **Modify IAM role** → **AmazonEC2ContainerRegistryReadOnly**

### Install Docker + nginx

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y git nginx certbot python3-certbot-nginx docker.io docker-compose-v2

sudo usermod -aG docker ubuntu
```

Log out and SSH back in:

```bash
docker --version
docker compose version    # space — NOT docker-compose
```

If `docker compose` missing: `sudo apt install -y docker-compose-v2`

Reboot if **“System restart required”**: `sudo reboot`

### Clone and configure env

```bash
git clone https://github.com/ankit-devwork/insight-lab.git
cd insight-lab

cp backend/.env.example backend/.env
cp .env.ecr.example .env.ecr
nano backend/.env
nano .env.ecr
```

**`.env.ecr`:**

```bash
ECR_REGISTRY=321204595484.dkr.ecr.us-east-1.amazonaws.com
ECR_REPOSITORY=insight-lab
API_IMAGE_TAG=api-latest
AWS_REGION=us-east-1
NEO4J_AUTH=neo4j/insightlab_dev_password
```

**`backend/.env` (minimum):**

```bash
APP_APP__ENV=production
APP_DEBUG=false

SUPABASE_URL=https://xxxx.supabase.co
SUPABASE_SERVICE_ROLE_KEY=...
SUPABASE_JWT_SECRET=...

NEO4J_USER=neo4j
NEO4J_PASSWORD=insightlab_dev_password

GROQ_API_KEY=gsk_...

# Optional with Vercel proxy (browser hits Vercel, not EC2 directly)
CORS_ALLOW_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
```

### If `docker-compose.ecr.yml` is missing

After clone, run `ls docker-compose.ecr.yml`. If missing, `git pull origin main` or copy the file from the repo. The ECR compose file defines **api + redis + neo4j** together.

### Free port 8000

```bash
docker ps
# If genai or other stack uses 8000:
docker stop genai_backend genai_streamlit
```

### Pull and start (full stack)

```bash
cd ~/insight-lab

aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com

docker compose -f docker-compose.ecr.yml \
  --env-file backend/.env \
  --env-file .env.ecr \
  pull

docker compose -f docker-compose.ecr.yml \
  --env-file backend/.env \
  --env-file .env.ecr \
  up -d
```

Always pass **both** env files. Without `.env.ecr`, `docker compose ps` warns `ECR_REGISTRY variable is not set`.

**Shell alias (recommended):**

```bash
echo "alias ilab='docker compose -f ~/insight-lab/docker-compose.ecr.yml --env-file ~/insight-lab/backend/.env --env-file ~/insight-lab/.env.ecr'" >> ~/.bashrc
source ~/.bashrc
```

Then: `ilab ps`, `ilab logs -f api`, `ilab up -d`

### Verify backend

```bash
ilab ps
# Expect: insightlab-api, insightlab-redis, insightlab-neo4j — all healthy

curl -s http://127.0.0.1:8000/health    # "status":"ok"
curl -s http://127.0.0.1:8000/ready     # "status":"ready"
```

**Only redis + neo4j?** You ran `docker compose up -d` without `-f docker-compose.ecr.yml`.

### nginx on port 8080 (for Vercel)

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

Test from EC2 and Windows:

```bash
curl -s http://127.0.0.1:8080/health
```

```powershell
curl http://YOUR_EC2_PUBLIC_IP:8080/health
```

### Backend logs

```bash
ilab logs -f api
ilab logs --tail 100 api
docker exec insightlab-api tail -f /app/logs/*.log
```

---

## Part 3 — Deploy API updates

**Windows** (after backend code changes):

```powershell
.\scripts\push-ecr.ps1
```

**EC2:**

```bash
cd ~/insight-lab
ilab pull
ilab up -d
```

---

## Part 4 — Vercel frontend

### Import project correctly

Vercel → **Add New Project** → import `ankit-devwork/insight-lab`

| Setting | Value |
|---------|-------|
| **Root Directory** | `frontend` |
| **Framework** | Next.js |

**Do not** use “Services” preset that deploys FastAPI on Vercel. Backend stays on EC2.

### Next.js version

Vercel **blocks** vulnerable Next.js (CVE-2025-66478). Use **≥ 15.5.18**:

```powershell
cd frontend
npm install next@15.5.18 eslint-config-next@15.5.18
```

Commit and push before deploying.

### API proxy (no domain)

The repo includes `frontend/next.config.ts` rewrite:

```text
Browser → https://your-app.vercel.app/api-backend/*
       → Vercel server (BACKEND_PROXY_URL)
       → http://EC2-IP:8080/*
       → nginx → Docker API
```

Browsers cannot call `http://EC2-IP` directly from an HTTPS Vercel page (mixed content). The proxy fixes this.

### Vercel environment variables

Settings → Environment Variables → **Production**:

| Variable | Value |
|----------|-------|
| `NEXT_PUBLIC_SUPABASE_URL` | `https://xxxx.supabase.co` |
| `NEXT_PUBLIC_SUPABASE_ANON_KEY` | anon key |
| `NEXT_PUBLIC_API_URL` | `/api-backend` |
| `BACKEND_PROXY_URL` | `http://YOUR_EC2_PUBLIC_IP:8080` |
| `NEXT_PUBLIC_SHOW_DEV_PANEL` | `false` |

- `BACKEND_PROXY_URL` has **no** `NEXT_PUBLIC_` prefix
- Use `http://` and port **8080**
- **Redeploy** after adding or changing env vars

### Verify proxy

```text
https://YOUR-APP.vercel.app/api-backend/health
```

Expected: `{"status":"ok",...}`

### Supabase Auth (fix localhost redirect)

If signup/login redirects to **localhost**, Supabase Site URL is wrong.

Authentication → **URL configuration**:

| Field | Value |
|-------|-------|
| **Site URL** | `https://YOUR-APP.vercel.app` |
| **Redirect URLs** | `https://YOUR-APP.vercel.app/**` |
| | `http://localhost:3000/**` (keep for local dev) |

Use incognito to test after saving. Old confirmation emails may still point to localhost — sign up again or sign in directly.

### Signup confirmation email not received

InsightLab uses **Supabase Auth** for email signup. The app cannot send those emails itself — Supabase (or your custom SMTP) does.

**Quick checks (Supabase dashboard):**

1. **Authentication → Users** — find `ankitsrivastava4u@gmail.com`. If the user exists but is unconfirmed, use **Confirm user** (three-dot menu) to activate immediately, then sign in.
2. **Authentication → Logs** — look for `user.signup` / mail events and any SMTP errors.
3. **Authentication → URL configuration** — **Site URL** must be your Vercel URL; **Redirect URLs** must include `https://YOUR-APP.vercel.app/**`.
4. **Gmail** — check **Spam** and **Promotions**. Supabase’s default sender often lands there.

**If emails never arrive (common on free tier):**

Supabase’s built-in mail has low volume and deliverability. For production, configure custom SMTP:

**Project Settings → Authentication → SMTP Settings** → enable custom SMTP (Resend, SendGrid, AWS SES, etc.).

Or, for internal testing only: **Authentication → Providers → Email** → disable **Confirm email** (users can sign in immediately after signup).

The signup and login pages include **Resend confirmation email** after signup or when login fails with “Email not confirmed”.

### Upload size note

Vercel Hobby may limit proxied uploads to ~**4.5 MB**. InsightLab allows **10 MB**. If uploads fail on Vercel but work via `curl` to EC2, use HTTPS API (domain / Cloudflare Tunnel) or host frontend on EC2.

---

## Part 5 — Optional paths

### Custom domain + HTTPS API

Skip the Vercel proxy. Point `api.yourdomain.com` → EC2 nginx on **443**, set:

```bash
NEXT_PUBLIC_API_URL=https://api.yourdomain.com
CORS_ALLOW_ORIGINS=https://your-app.vercel.app
```

See [DEPLOY-EC2.md — nginx](DEPLOY-EC2.md#6-nginx-reverse-proxy).

### Frontend on EC2 (no Vercel)

Run Next.js on EC2 port 3000 + nginx port 80. No mixed-content issue. See discussion in team docs or host both on same IP (80 → frontend, 8080 → API).

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `docker-compose-plugin` not found | `sudo apt install docker.io docker-compose-v2` |
| `docker-compose: command not found` | Use **`docker compose`** (space) |
| `open docker-compose.ecr.yml: no such file` | `git pull origin main` or copy file from repo |
| Only redis + neo4j, no API | Use `-f docker-compose.ecr.yml` with env files |
| `python3-venv` error | Skip — no Python on EC2 for ECR path |
| Port 8000 in use | `docker ps` → stop other stacks |
| `Cannot pull image` | IAM **AmazonEC2ContainerRegistryReadOnly** + ECR login |
| Vercel build: vulnerable Next.js | Upgrade to **15.5.18+** |
| Vercel shows frontend + FastAPI | Root directory = **`frontend`** only |
| Redirect to localhost after auth | Supabase **Site URL** = Vercel URL |
| No signup confirmation email | Supabase **Auth logs**; confirm user manually; configure **SMTP**; check spam |
| Login: “Email not confirmed” | Use **Resend confirmation email** on login, or confirm user in Supabase |
| `/api-backend/health` fails | EC2 **8080** open? nginx running? `BACKEND_PROXY_URL` correct? |
| Upload fails on Vercel only | ~4.5 MB proxy limit |
| `/ready` 503 | `ilab ps` — check redis/neo4j; `ilab logs api` |

---

## Command cheat sheet

```bash
# EC2 — always use ilab alias or full compose command with both env files
ilab ps
ilab logs -f api
ilab pull && ilab up -d

curl -s http://127.0.0.1:8000/health
curl -s http://EC2_PUBLIC_IP:8080/health
```

```powershell
# Windows — push new API image
.\scripts\push-ecr.ps1
```

```text
# Vercel — test proxy
https://YOUR-APP.vercel.app/api-backend/health
```

| Wrong | Correct |
|-------|---------|
| `docker compose up -d` | `docker compose -f docker-compose.ecr.yml --env-file backend/.env --env-file .env.ecr up -d` |
| Deploy backend on Vercel | Backend on EC2 only |
| `NEXT_PUBLIC_API_URL=http://EC2-IP:8080` | `NEXT_PUBLIC_API_URL=/api-backend` + `BACKEND_PROXY_URL` |
| Supabase Site URL = localhost | Site URL = Vercel URL |

---

## Related docs

- [DEPLOY-EC2.md](DEPLOY-EC2.md) — manual venv deploy (fallback)
- [frontend/README.md](../frontend/README.md) — local dev + Vercel env
- [ARCHITECTURE.md](ARCHITECTURE.md) — system design
