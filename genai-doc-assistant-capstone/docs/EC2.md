# Deploy to AWS EC2 (ECR + Docker Compose)

Run the **backend** and **Streamlit** containers on a single EC2 instance using images stored in **Amazon ECR**.

This guide reflects a **verified deployment** using one ECR repository (`digital-worker-studio`) and two image tags.

## Architecture

```text
Your laptop                         AWS
───────────                         ───
build + push ──► ECR repo             EC2 instance
  digital-worker-studio:              ├── docker compose (ecr.yml)
    ├── genai-backend-latest          ├── backend :8000
    └── genai-streamlit-latest        └── streamlit :8501
                                          └── BACKEND_URL=http://backend:8000
```

| Component | Role |
|-----------|------|
| **ECR** | One repo, two tags (backend + Streamlit) |
| **EC2** | Pulls images and runs `docker-compose.ecr.yml` |
| **Named volumes** | Persist uploads, Chroma vector DB, logs |
| **Groq API** | External LLM (set `GROQ_API_KEY` in `.env`) |

Users open **Streamlit** at `http://<ec2-public-ip>:8501`. Streamlit calls the backend on the internal Docker network (`http://backend:8000`). Port **8000** does not need to be public unless you want `/docs` exposed.

---

## Image naming (important)

Use **one ECR repository** and **different tags** per service:

| Service | ECR image URI |
|---------|----------------|
| Backend | `<account>.dkr.ecr.<region>.amazonaws.com/digital-worker-studio:genai-backend-latest` |
| Streamlit | `<account>.dkr.ecr.<region>.amazonaws.com/digital-worker-studio:genai-streamlit-latest` |

`docker-compose.ecr.yml` must reference:

```yaml
image: ${ECR_REGISTRY}/${ECR_REPOSITORY}:${BACKEND_IMAGE_TAG:-genai-backend-latest}   # backend
image: ${ECR_REGISTRY}/${ECR_REPOSITORY}:${STREAMLIT_IMAGE_TAG:-genai-streamlit-latest}  # streamlit
```

**Do not use** the old pattern (separate repos per service):

```yaml
# WRONG — will fail with "not found"
image: ${ECR_REGISTRY}/genai-doc-assistant-backend:${IMAGE_TAG:-latest}
image: ${ECR_REGISTRY}/genai-doc-assistant-streamlit:${IMAGE_TAG:-latest}
```

Verify resolved URLs before pull:

```bash
sudo docker compose -f docker-compose.ecr.yml --env-file .env --env-file .env.ecr config | grep image:
```

---

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| AWS account | ECR + EC2 |
| AWS CLI | `aws configure` on laptop; IAM role or `aws configure` on EC2 |
| Docker + Compose | Laptop (build/push) and EC2 (run) |
| Groq API key | [console.groq.com](https://console.groq.com) |

### Recommended EC2 sizing

| Resource | Recommendation |
|----------|----------------|
| Instance | `t3.medium` (4 GB RAM) minimum; `t3.large` (8 GB) for default embedding model |
| AMI | Ubuntu 22.04 LTS |
| EBS root volume | **20–30 GB** minimum (default ~8 GB is too small for Docker + images) |
| Security group | Inbound: **22** (SSH, your IP), **8501** (Streamlit). Optional: **8000** (API `/docs`) |

---

## Part 1 — Build and push images (your Windows machine)

### 1. Get AWS account ID

```powershell
aws sts get-caller-identity --query Account --output text
```

Example: `321204595484` → registry `321204595484.dkr.ecr.us-east-1.amazonaws.com`

### 2. Create ECR repo (once)

```powershell
aws ecr create-repository --repository-name digital-worker-studio --region us-east-1
```

### 3. Build, tag, and push both images

From monorepo root (`the-learning-curve-labs/`):

```powershell
cd D:\Mine\Learining\GenAI\python\the-learning-curve-labs

$ACCOUNT_ID = aws sts get-caller-identity --query Account --output text
$ECR_REGISTRY = "$ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com"
$ECR_REPO = "digital-worker-studio"
$REGION = "us-east-1"

aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ECR_REGISTRY

docker build -f genai-doc-assistant-capstone/Dockerfile -t genai-doc-assistant-backend:local .
docker build -f genai-doc-assistant-capstone/front-end/streamlit/Dockerfile -t genai-doc-assistant-streamlit:local .

docker tag genai-doc-assistant-backend:local "${ECR_REGISTRY}/${ECR_REPO}:genai-backend-latest"
docker tag genai-doc-assistant-streamlit:local "${ECR_REGISTRY}/${ECR_REPO}:genai-streamlit-latest"

docker push "${ECR_REGISTRY}/${ECR_REPO}:genai-backend-latest"
docker push "${ECR_REGISTRY}/${ECR_REPO}:genai-streamlit-latest"
```

Confirm tags in ECR:

```bash
aws ecr describe-images --repository-name digital-worker-studio --region us-east-1 \
  --query 'imageDetails[*].imageTags' --output table
```

Expected: `genai-backend-latest`, `genai-streamlit-latest`

**Bash alternative** (Git Bash / WSL):

```bash
ECR_REGISTRY=123456789012.dkr.ecr.us-east-1.amazonaws.com \
ECR_REPOSITORY=digital-worker-studio \
AWS_REGION=us-east-1 \
./genai-doc-assistant-capstone/scripts/push-ecr.sh
```

---

## Part 2 — Launch and prepare EC2

### 1. SSH from Windows (fix PEM permissions if needed)

If you see `UNPROTECTED PRIVATE KEY FILE` or `bad permissions`:

```powershell
cd D:\Mine\Learining\AWS
icacls .\digital-worker-pem.pem /inheritance:r
icacls .\digital-worker-pem.pem /grant:r "$($env:USERNAME):(R)"
icacls .\digital-worker-pem.pem /remove "Authenticated Users"
icacls .\digital-worker-pem.pem /remove "Users"
icacls .\digital-worker-pem.pem /remove "Everyone"
```

Then connect:

```powershell
ssh -i digital-worker-pem.pem ubuntu@<ec2-public-ip>
```

### 2. Expand disk (if root is ~7 GB and full)

Check:

```bash
df -h /
lsblk
```

If `xvda` is 20G but `xvda1` is only ~7G, grow the partition:

```bash
sudo apt install -y cloud-guest-utils
sudo growpart /dev/xvda 1
sudo resize2fs /dev/xvda1
df -h /
```

If the EBS volume is still small, expand it in **EC2 → Volumes → Modify volume** (set 20–30 GB), then run `growpart` + `resize2fs` again.

### 3. Install Docker

```bash
sudo apt update
sudo apt install -y docker.io docker-compose-v2 awscli
sudo usermod -aG docker ubuntu
```

Exit and SSH back in, then verify:

```bash
docker --version
docker compose version
df -h /
```

### 4. IAM role for ECR pull

Attach IAM role **`AmazonEC2ContainerRegistryReadOnly`** to the EC2 instance (recommended). Otherwise run `aws configure` on the instance.

### 5. Security group

Add inbound rules:

| Port | Purpose |
|------|---------|
| 22 | SSH |
| **8501** | Streamlit UI (required for browser access) |
| 8000 | Optional — FastAPI `/docs` |

> Port 8000 can work while 8501 is blocked — open **8501** explicitly for the UI.

---

## Part 3 — Deploy on EC2

### 1. Create app directory

```bash
mkdir -p ~/genai-app
cd ~/genai-app
```

### 2. Copy `docker-compose.ecr.yml` from laptop

```powershell
scp -i D:\Mine\Learining\AWS\digital-worker-pem.pem `
  D:\Mine\Learining\GenAI\python\the-learning-curve-labs\genai-doc-assistant-capstone\docker-compose.ecr.yml `
  ubuntu@<ec2-public-ip>:~/genai-app/
```

Confirm image lines on EC2:

```bash
grep "image:" ~/genai-app/docker-compose.ecr.yml
```

Must show `${ECR_REPOSITORY}` and `${BACKEND_IMAGE_TAG}` / `${STREAMLIT_IMAGE_TAG}` — not `genai-doc-assistant-backend`.

**Quick fix** if you have the old file:

```bash
cd ~/genai-app
sed -i 's|image: ${ECR_REGISTRY}/genai-doc-assistant-backend:${IMAGE_TAG:-latest}|image: ${ECR_REGISTRY}/${ECR_REPOSITORY}:${BACKEND_IMAGE_TAG:-genai-backend-latest}|' docker-compose.ecr.yml
sed -i 's|image: ${ECR_REGISTRY}/genai-doc-assistant-streamlit:${IMAGE_TAG:-latest}|image: ${ECR_REGISTRY}/${ECR_REPOSITORY}:${STREAMLIT_IMAGE_TAG:-genai-streamlit-latest}|' docker-compose.ecr.yml
```

### 3. Create `.env`

```bash
nano ~/genai-app/.env
```

```env
GROQ_API_KEY=gsk_your_key_here
APP_ENV=production
LANGCHAIN_TRACING_V2=false
```

### 4. Create `.env.ecr`

```bash
nano ~/genai-app/.env.ecr
```

```env
ECR_REGISTRY=321204595484.dkr.ecr.us-east-1.amazonaws.com
ECR_REPOSITORY=digital-worker-studio
AWS_REGION=us-east-1
BACKEND_IMAGE_TAG=genai-backend-latest
STREAMLIT_IMAGE_TAG=genai-streamlit-latest
```

Replace `321204595484` with your account ID.

### 5. ECR login, pull, and start

Use `sudo` consistently if you logged in with `sudo docker login`:

```bash
cd ~/genai-app
set -a && source .env.ecr && set +a

aws ecr get-login-password --region $AWS_REGION | \
  sudo docker login --username AWS --password-stdin $ECR_REGISTRY

sudo docker compose -f docker-compose.ecr.yml --env-file .env --env-file .env.ecr config | grep image:

sudo docker compose -f docker-compose.ecr.yml --env-file .env --env-file .env.ecr pull
sudo docker compose -f docker-compose.ecr.yml --env-file .env --env-file .env.ecr up -d
```

### 6. Verify

```bash
sudo docker compose -f docker-compose.ecr.yml --env-file .env --env-file .env.ecr ps
curl http://localhost:8000/health
curl http://localhost:8501/_stcore/health
```

Browser:

- Streamlit UI: `http://<ec2-public-ip>:8501`
- API docs (optional): `http://<ec2-public-ip>:8000/docs`

---

## Operations

### View logs

```bash
cd ~/genai-app
COMPOSE="sudo docker compose -f docker-compose.ecr.yml --env-file .env --env-file .env.ecr"

$COMPOSE logs backend --tail 50
$COMPOSE logs streamlit --tail 50
$COMPOSE logs -f backend          # follow live
```

Or directly:

```bash
sudo docker logs genai_backend --tail 50
sudo docker logs genai_streamlit --tail 50
```

### Restart / stop

```bash
sudo docker compose -f docker-compose.ecr.yml --env-file .env --env-file .env.ecr restart
sudo docker compose -f docker-compose.ecr.yml --env-file .env --env-file .env.ecr down
```

### Update after new image push

**Laptop:** rebuild and push with new tags (or reuse `genai-backend-latest`).

**EC2:**

```bash
cd ~/genai-app
set -a && source .env.ecr && set +a
aws ecr get-login-password --region $AWS_REGION | sudo docker login --username AWS --password-stdin $ECR_REGISTRY
sudo docker compose -f docker-compose.ecr.yml --env-file .env --env-file .env.ecr pull
sudo docker compose -f docker-compose.ecr.yml --env-file .env --env-file .env.ecr up -d
```

---

## Environment variables

| Variable | Where | Required | Description |
|----------|-------|----------|-------------|
| `GROQ_API_KEY` | `.env` | Yes | LLM API key |
| `ECR_REGISTRY` | `.env.ecr` | Yes | ECR registry host |
| `ECR_REPOSITORY` | `.env.ecr` | Yes | ECR repo name (`digital-worker-studio`) |
| `BACKEND_IMAGE_TAG` | `.env.ecr` | No | Default `genai-backend-latest` |
| `STREAMLIT_IMAGE_TAG` | `.env.ecr` | No | Default `genai-streamlit-latest` |
| `BACKEND_URL` | compose | Auto | `http://backend:8000` inside Docker network |

Always pass both env files to compose:

```bash
sudo docker compose -f docker-compose.ecr.yml --env-file .env --env-file .env.ecr <command>
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `genai-doc-assistant-streamlit:latest: not found` | Old `docker-compose.ecr.yml` or wrong `.env.ecr` | Use `${ECR_REPOSITORY}` + tag vars; run `config \| grep image:` |
| `no basic auth credentials` | `sudo docker login` but compose without `sudo` | Use `sudo` for both login and compose |
| `ECR_REGISTRY variable is not set` warning | `ps`/`logs` without `--env-file .env.ecr` | Add `--env-file .env --env-file .env.ecr` or `source .env.ecr` |
| `apt install docker` fails, disk full | Root volume too small | Expand EBS; `growpart` + `resize2fs` |
| `:8000/docs` works, `:8501` does not | Security group | Add inbound **TCP 8501** |
| SSH `bad permissions` on `.pem` (Windows) | ACL too open | `icacls` steps in Part 2 |
| OOM on upload/query | Instance too small | `t3.large` or MiniLM embedding model in `.env` |
| Slow first query | Model download | Normal on first use |

### Verify ECR tags

```bash
aws ecr describe-images --repository-name digital-worker-studio --region us-east-1 \
  --query 'imageDetails[*].imageTags' --output table
```

### Reset all app data

```bash
sudo docker compose -f docker-compose.ecr.yml --env-file .env --env-file .env.ecr down -v
```

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
