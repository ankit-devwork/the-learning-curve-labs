# Deploy InsightLab with ECR + EC2

**Recommended production pilot:** build the API image locally, push to **Amazon ECR**, pull and run on a **single EC2** instance with nginx. Frontend stays on **Vercel**; Supabase stays hosted.

**Repository:** [github.com/ankit-devwork/insight-lab](https://github.com/ankit-devwork/insight-lab)

For the older git-clone + venv path, see [DEPLOY-EC2.md](DEPLOY-EC2.md).

---

## Architecture

```text
Windows (local)                         AWS
───────────────                         ───
docker build  ──►  ECR                  EC2
docker push         insight-lab:api-latest   ├── docker compose (ecr.yml)
                                             │     ├── api (ECR image) :8000 localhost
                                             │     ├── redis
                                             │     └── neo4j
                                             └── nginx :443 → localhost:8000

Vercel (Next.js) ──► Supabase + EC2 API
```

| Component | Where |
|-----------|-------|
| Frontend | Vercel |
| API image | ECR → Docker on EC2 |
| Redis + Neo4j | Docker on same EC2 (`docker-compose.ecr.yml`) |
| Auth, Postgres, Storage | Supabase (hosted) |
| TLS | nginx + Certbot on EC2 |

---

## Cost: EC2 vs “Fleet” vs Fargate

People often say **“Fleet”** when they mean one of these:

| Option | What it is | Cost for InsightLab pilot |
|--------|------------|---------------------------|
| **Single EC2** (t3.small/medium) | One always-on VM | **Cheapest and simplest** (~$15–30/mo + EBS) |
| **EC2 Spot / Spot Fleet** | Same as EC2, discounted spare capacity | ~60–70% cheaper; instance can be **interrupted** |
| **ECS on Fargate** | Serverless containers, no VM to patch | Usually **more expensive** for 24/7 API + you still need Neo4j somewhere |
| **Lightsail** | Fixed-price small VPS | Often **cheapest** predictable bill ($10–20/mo tier) |

**Recommendation for InsightLab:**

- Use **ECR for the API image** (repeatable deploys).
- Run containers on **one small EC2** (or Lightsail with Docker installed).
- Do **not** move to Fargate yet — Neo4j + Redis + FastEmbed fit better on one box.
- Consider **Spot** only after the pilot is stable and you accept occasional restarts.

ECR storage is cheap (~$0.10/GB/month). One API image is typically well under $1/mo.

Optional savings: use **Upstash** for Redis (free tier) and drop the local Redis container — Neo4j still needs to run on EC2 for adaptive quiz / graph sync.

---

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| AWS account | ECR + EC2 |
| AWS CLI | `aws configure` on Windows; IAM role on EC2 |
| Docker Desktop | Windows — build and push |
| Docker on EC2 | Pull and run |
| Supabase | Migrations **001–007** applied |
| Groq API key | LLM provider |
| Security group | Inbound: **22** (My IP), **80**, **443**. **Do not** open **8000** publicly |

### Recommended EC2 sizing

| Resource | Recommendation |
|----------|----------------|
| Instance | **t3.medium** (4 GB RAM) — FastEmbed + PDF processing |
| AMI | Ubuntu 22.04 LTS |
| EBS | **20–30 GB** (Docker images + Neo4j + logs) |

---

## Part 1 — Build and push (Windows)

From your `insight-lab` clone:

```powershell
cd D:\Mine\Learining\GenAI\python\insight-lab

# One-time AWS setup
aws configure

# Build + push (creates ECR repo if missing)
.\scripts\push-ecr.ps1
```

Or set variables explicitly:

```powershell
$env:AWS_REGION = "us-east-1"
$env:ECR_REPOSITORY = "insight-lab"
$env:API_IMAGE_TAG = "api-latest"
.\scripts\push-ecr.ps1
```

Linux/macOS:

```bash
export ECR_REGISTRY="$(aws sts get-caller-identity --query Account --output text).dkr.ecr.us-east-1.amazonaws.com"
export ECR_REPOSITORY=insight-lab
./scripts/push-ecr.sh
```

---

## Part 2 — EC2 setup (once)

### 1. Launch EC2 + security group

Same as [DEPLOY-EC2.md](DEPLOY-EC2.md) section 1 — inbound **22** (My IP), **80**, **443**.

Attach an **IAM role** to the instance with **AmazonEC2ContainerRegistryReadOnly** so EC2 can pull from ECR without storing long-lived keys.

### 2. Install Docker + nginx

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y docker.io docker-compose-plugin nginx certbot python3-certbot-nginx git

sudo usermod -aG docker $USER
# log out and back in
```

### 3. Clone repo (for compose files + env only — not for running Python on host)

```bash
git clone https://github.com/ankit-devwork/insight-lab.git
cd insight-lab

cp backend/.env.example backend/.env
cp .env.ecr.example .env.ecr
# Edit backend/.env — Supabase, Groq, CORS, APP_APP__ENV=production
# Edit .env.ecr — your ECR_REGISTRY and account ID
```

**Important:** In `backend/.env`, set container-friendly hosts (compose overrides these, but keep them consistent):

```bash
REDIS_HOST=redis
NEO4J_URI=bolt://neo4j:7687
CORS_ALLOW_ORIGINS=https://app.yourdomain.com
APP_APP__ENV=production
```

### 4. Pull and start

```bash
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com

docker compose -f docker-compose.ecr.yml \
  --env-file backend/.env \
  --env-file .env.ecr \
  pull

docker compose -f docker-compose.ecr.yml \
  --env-file backend/.env \
  --env-file .env.ecr \
  up -d

curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/ready
```

### 5. nginx (same as DEPLOY-EC2)

Proxy `api.yourdomain.com` → `http://127.0.0.1:8000` with 300s timeouts for document processing. See [DEPLOY-EC2.md — nginx](DEPLOY-EC2.md#6-nginx-reverse-proxy).

---

## Part 3 — Deploy updates

**On Windows** after code changes:

```powershell
.\scripts\push-ecr.ps1
# Optional: bump tag, e.g. $env:API_IMAGE_TAG = "api-2025-06-12"
```

**On EC2:**

```bash
cd ~/insight-lab
docker compose -f docker-compose.ecr.yml --env-file backend/.env --env-file .env.ecr pull
docker compose -f docker-compose.ecr.yml --env-file backend/.env --env-file .env.ecr up -d
```

Rollback: push/previous tag in ECR, update `API_IMAGE_TAG` in `.env.ecr`, pull and up again.

---

## Part 4 — Frontend (Vercel)

Unchanged from [DEPLOY-EC2.md](DEPLOY-EC2.md):

```bash
NEXT_PUBLIC_SUPABASE_URL=https://xxxx.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=eyJ...
NEXT_PUBLIC_API_URL=https://api.yourdomain.com
NEXT_PUBLIC_SHOW_DEV_PANEL=false
```

Configure Supabase Auth Site URL + redirect URLs for your Vercel domain.

---

## Smoke test

| Step | Verify |
|------|--------|
| `curl https://api.yourdomain.com/health` | `{"status":"ok",...}` |
| Login on Vercel app | Dashboard loads |
| Upload PDF | Processes, summary appears |
| Ask / Quiz / Multi-doc | End-to-end works |

Logs:

```bash
docker compose -f docker-compose.ecr.yml logs -f api
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `Cannot pull image` on EC2 | IAM role **AmazonEC2ContainerRegistryReadOnly**; run `docker login` to ECR |
| API exits on start | `docker compose logs api` — usually missing env in `backend/.env` |
| `/ready` 503 | Redis or Neo4j unhealthy — `docker compose ps` |
| CORS errors | `CORS_ALLOW_ORIGINS` must match Vercel URL exactly |
| OOM / slow first request | Use t3.medium; first FastEmbed download takes time |
| Build fails on pycorekit | Ensure Docker build has network access for GitHub pip install |

---

## Related docs

- [DEPLOY-EC2.md](DEPLOY-EC2.md) — manual venv deploy (fallback)
- [ARCHITECTURE.md](ARCHITECTURE.md) — system design
- [backend/README.md](../backend/README.md) — API and env reference
