# Deploy to AWS EC2 (ECR + Docker Compose)

Run the **backend** and **Streamlit** containers on a single EC2 instance using images stored in **Amazon ECR**.

## Architecture

```text
Your laptop                         AWS
───────────                         ───
build + push ──► ECR repos          EC2 instance
  ├── genai-doc-assistant-backend       ├── docker compose (ecr.yml)
  └── genai-doc-assistant-streamlit     ├── backend :8000
                                        └── streamlit :8501
                                              └── BACKEND_URL=http://backend:8000
```

| Component | Role |
|-----------|------|
| **ECR** | Stores both Docker images |
| **EC2** | Pulls images and runs `docker-compose.ecr.yml` |
| **Named volumes** | Persist uploads, Chroma vector DB, logs |
| **Groq API** | External LLM (set `GROQ_API_KEY` in `.env`) |

Users open **Streamlit** at `http://<ec2-public-ip>:8501`. Streamlit calls the backend over the internal Docker network — port 8000 does not need to be public unless you want direct API access.

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| AWS account | ECR + EC2 |
| AWS CLI | Configured locally (`aws configure`) and on EC2 |
| Docker + Compose | On your machine (build/push) and EC2 (run) |
| Groq API key | [console.groq.com](https://console.groq.com) |

### Recommended EC2 sizing

| Resource | Recommendation |
|----------|----------------|
| Instance | `t3.medium` (4 GB RAM) minimum; `t3.large` (8 GB) for default embedding model |
| AMI | Ubuntu 22.04 LTS |
| EBS | 30 GB gp3 |
| Security group | Inbound: 22 (SSH, your IP), 8501 (Streamlit). Optional: 8000 (API) |

---

## Part 1 — Build and push images (your machine)

### 1. Create ECR repositories (once)

```bash
aws ecr create-repository --repository-name genai-doc-assistant-backend --region us-east-1
aws ecr create-repository --repository-name genai-doc-assistant-streamlit --region us-east-1
```

Or let `scripts/push-ecr.sh` create them automatically.

### 2. Push both images

From the **monorepo root** (`the-learning-curve-labs/`):

```bash
chmod +x genai-doc-assistant-capstone/scripts/push-ecr.sh

ECR_REGISTRY=123456789012.dkr.ecr.us-east-1.amazonaws.com \
AWS_REGION=us-east-1 \
IMAGE_TAG=latest \
./genai-doc-assistant-capstone/scripts/push-ecr.sh
```

**PowerShell (manual steps):**

```powershell
$ECR_REGISTRY = "123456789012.dkr.ecr.us-east-1.amazonaws.com"
$REGION = "us-east-1"

aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ECR_REGISTRY

docker build -f genai-doc-assistant-capstone/Dockerfile -t genai-doc-assistant-backend:local .
docker build -f genai-doc-assistant-capstone/front-end/streamlit/Dockerfile -t genai-doc-assistant-streamlit:local .

docker tag genai-doc-assistant-backend:local "$ECR_REGISTRY/genai-doc-assistant-backend:latest"
docker tag genai-doc-assistant-streamlit:local "$ECR_REGISTRY/genai-doc-assistant-streamlit:latest"

docker push "$ECR_REGISTRY/genai-doc-assistant-backend:latest"
docker push "$ECR_REGISTRY/genai-doc-assistant-streamlit:latest"
```

You must push **both** images — backend and Streamlit are separate containers.

---

## Part 2 — Run on EC2

### 1. Launch and connect

```bash
ssh -i your-key.pem ubuntu@<ec2-public-ip>
```

### 2. Install Docker

```bash
sudo apt update
sudo apt install -y docker.io docker-compose-v2 awscli
sudo usermod -aG docker ubuntu
# Log out and back in so the docker group applies
```

### 3. IAM permissions for ECR pull

Attach an IAM role to the EC2 instance (recommended) with `AmazonEC2ContainerRegistryReadOnly`, or configure `aws configure` on the instance.

### 4. Copy deployment files to EC2

Copy these files into e.g. `~/genai-app/` on the instance:

| File | Purpose |
|------|---------|
| `docker-compose.ecr.yml` | Pull-and-run stack |
| `.env` | Secrets (`GROQ_API_KEY`, etc.) |
| `.env.ecr` | `ECR_REGISTRY`, `IMAGE_TAG` |

```bash
# On EC2
mkdir -p ~/genai-app
cd ~/genai-app

cp .env.example .env    # if you copied the example from the repo
# Edit .env — set GROQ_API_KEY at minimum

cp .env.ecr.example .env.ecr
# Edit .env.ecr — set your ECR_REGISTRY
```

Example `.env`:

```bash
GROQ_API_KEY=gsk_your_key_here
APP_ENV=production
LANGCHAIN_TRACING_V2=false
```

Example `.env.ecr`:

```bash
ECR_REGISTRY=123456789012.dkr.ecr.us-east-1.amazonaws.com
AWS_REGION=us-east-1
IMAGE_TAG=latest
```

### 5. Log in to ECR and start

```bash
cd ~/genai-app

aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin 123456789012.dkr.ecr.us-east-1.amazonaws.com

docker compose -f docker-compose.ecr.yml --env-file .env --env-file .env.ecr pull
docker compose -f docker-compose.ecr.yml --env-file .env --env-file .env.ecr up -d
```

### 6. Verify

```bash
docker compose -f docker-compose.ecr.yml ps
curl http://localhost:8000/health
curl http://localhost:8501/_stcore/health
```

Open `http://<ec2-public-ip>:8501` in your browser.

---

## Updates (new code release)

**On your machine:**

```bash
IMAGE_TAG=v1.0.1 ./genai-doc-assistant-capstone/scripts/push-ecr.sh
```

**On EC2:**

```bash
# Update IMAGE_TAG in .env.ecr if you used a version tag
docker compose -f docker-compose.ecr.yml --env-file .env --env-file .env.ecr pull
docker compose -f docker-compose.ecr.yml --env-file .env --env-file .env.ecr up -d
```

Data in named volumes (`genai-uploads`, `genai-vector-store`) survives image updates.

---

## Environment variables

| Variable | Where | Required | Description |
|----------|-------|----------|-------------|
| `GROQ_API_KEY` | `.env` | Yes | LLM API key |
| `ECR_REGISTRY` | `.env.ecr` | Yes | ECR registry host |
| `IMAGE_TAG` | `.env.ecr` | No | Image tag (default `latest`) |
| `BACKEND_URL` | compose | Auto | `http://backend:8000` inside Docker network |
| `APP_ENV` | compose | Auto | Set to `production` in `docker-compose.ecr.yml` |
| `CORS_ALLOW_ORIGINS` | `.env` | No | Defaults to `*`; tighten if exposing API publicly |

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `no basic auth credentials` on pull | Run `aws ecr get-login-password \| docker login ...` on EC2 |
| Streamlit cannot reach API | Ensure `BACKEND_URL=http://backend:8000` (set in `docker-compose.ecr.yml`) |
| OOM on upload/query | Use `t3.large` or set `APP_MODELS__EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2` |
| Slow first query | Embedding model downloads on first use — normal |
| Reset all data | `docker compose -f docker-compose.ecr.yml down -v` |

---

## Optional: HTTPS with a domain

1. Point a domain A-record to the EC2 public IP.
2. Install Nginx or Caddy on the host.
3. Reverse-proxy port 443 → `localhost:8501`.
4. Use Let's Encrypt for TLS.

---

## Related docs

- [DOCKER.md](DOCKER.md) — local build and dev compose
- [FREE_DEPLOY.md](FREE_DEPLOY.md) — local demo + tunnel options
- [CONFIGURATION.md](CONFIGURATION.md) — `.env` overrides
