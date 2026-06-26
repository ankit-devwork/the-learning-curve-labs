# GenAI Document Assistant — Shared Folder Setup Guide

Use this guide when the project is copied to a folder such as `D:\Mine\Learining\CapstoneProject`.

---

## 1. Required folder layout

The backend **depends on `pycorekit`**. Copy **both** folders into the same parent directory:

```text
CapstoneProject/                    ← parent (your shared root)
├── README.md                       ← copy this file here (optional)
├── pycorekit/                      ← REQUIRED — shared library
└── genai-doc-assistant-capstone/   ← main application
    ├── app/
    ├── front-end/streamlit/
    ├── config.yaml
    ├── .env.example
    ├── docker-compose.yml
    ├── Dockerfile
    └── ...
```

**Do not share only `genai-doc-assistant-capstone/`** — Docker build will fail without `pycorekit/`.

---

## 2. Prerequisites (recipient machine)

| Requirement | Notes |
|-------------|-------|
| **Docker Desktop** | Windows 10/11, WSL2 enabled |
| **Groq API key** | Free at [console.groq.com](https://console.groq.com) |
| **RAM** | 8 GB+ recommended (embeddings use PyTorch) |
| **Disk** | ~10 GB free for Docker images |
| **Git** | Optional — not required if running from copied folder |

---

## 3. Quick start (Docker — recommended)

### Step 1 — Create `.env` from template

**PowerShell:**

```powershell
cd D:\Mine\Learining\CapstoneProject\genai-doc-assistant-capstone
copy .env.example .env
notepad .env
```

Set at minimum:

```env
GROQ_API_KEY=gsk_your_real_key_here
LANGCHAIN_TRACING_V2=false
```

**EC2 / low-RAM machines** — add for faster uploads:

```env
APP_RAG__SEMANTIC_DEDUPE=false
APP_MODELS__EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
APP_MODELS__EMBEDDING_DIM=384
```

> **Never share your real `.env` file.** Only share `.env.example`.

### Step 2 — Build and run

**From parent folder (`CapstoneProject`):**

```powershell
cd D:\Mine\Learining\CapstoneProject
docker compose -f genai-doc-assistant-capstone/docker-compose.yml up --build
```

**Or from capstone folder:**

```powershell
cd D:\Mine\Learining\CapstoneProject\genai-doc-assistant-capstone
docker compose up --build
```

First build may take **15–25 minutes** (downloads PyTorch + dependencies).

### Step 3 — Open the app

| Service | URL |
|---------|-----|
| **Streamlit UI** | http://localhost:8501 |
| **API docs** | http://localhost:8000/docs |
| **Health check** | http://localhost:8000/health |

### Step 4 — Try it

1. Upload a PDF in the sidebar (max **10 MB** per file).
2. Wait for ingestion (first run downloads the embedding model).
3. Ask a question in the chat box.

---

## 4. What to include when copying the folder

| Include | Exclude |
|---------|---------|
| `pycorekit/` (full source) | `.env` (contains secrets) |
| `genai-doc-assistant-capstone/` (full source) | `__pycache__/`, `.pytest_cache/` |
| `.env.example` | `data/uploads/*` (uploaded files) |
| `docs/` | `data/vector_store/*` (large Chroma data) |
| `config.yaml` | `logs/*` |
| Docker / compose files | `.git/` (optional) |

---

## 5. Configuration files (quick reference)

| File | Purpose |
|------|---------|
| `config.yaml` | Default settings (chunk size, models, 10 MB upload limit) |
| `.env.example` | Template — copy to `.env` and add `GROQ_API_KEY` |
| `.env` | **You create this** — secrets and overrides (not shared) |
| `docker-compose.yml` | Local Docker: builds and runs backend + Streamlit |
| `docker-compose.ecr.yml` | AWS EC2 only (pre-built images) |
| `.env.ecr.example` | AWS ECR settings (EC2 deploy only) |

Override example in `.env`:

```env
APP_FILE_UPLOAD__MAX_FILE_SIZE_MB=25
APP_MODELS__EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

See `docs/CONFIGURATION.md` for all options.

---

## 6. Shell scripts (what they do)

| Script | Used when |
|--------|-----------|
| `scripts/start-backend.sh` | Starts FastAPI inside backend Docker container |
| `front-end/streamlit/scripts/start-streamlit.sh` | Starts Streamlit; sets upload limit to match backend |
| `scripts/push-ecr.sh` | Push images to AWS ECR (cloud deploy only) |

Recipients running **local Docker only** do not need to run these scripts manually — Docker calls them.

---

## 7. Troubleshooting

| Problem | Fix |
|---------|-----|
| `pycorekit` not found during Docker build | Ensure `pycorekit/` is sibling of `genai-doc-assistant-capstone/` |
| `exec start-backend.sh: no such file or directory` | Rebuild with `--build`; use latest code (CRLF fix in Dockerfile) |
| LLM / query errors | Set valid `GROQ_API_KEY` in `.env`, restart containers |
| Upload shows 200 MB but fails at 10 MB | Rebuild Streamlit image from latest code (`/upload-limits` fix) |
| Port already in use | Stop other apps on 8000 / 8501 or change ports in compose |
| Slow first upload/query | Normal — embedding model downloads on first use (up to ~10 min on EC2) |
| Upload timed out after 300s | Set `LANGCHAIN_TRACING_V2=false`, `APP_RAG__SEMANTIC_DEDUPE=false`; restart; retry |
| LangSmith `429` warnings | Harmless — disable with `LANGCHAIN_TRACING_V2=false` in `.env` |
| Reset all data | `docker compose down -v` |

**Stop the app:**

```powershell
docker compose -f genai-doc-assistant-capstone/docker-compose.yml down
```

---

## 8. Optional: run without Docker (developers)

```powershell
cd D:\Mine\Learining\CapstoneProject\genai-doc-assistant-capstone
copy .env.example .env

pip install -e ..\pycorekit
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Terminal 1 — backend
uvicorn main:app --reload

# Terminal 2 — Streamlit
cd front-end\streamlit
set BACKEND_URL=http://127.0.0.1:8000
streamlit run chat.py
```

---

## 9. Optional: deploy to AWS EC2

See `genai-doc-assistant-capstone/docs/EC2.md` for full steps (ECR, EC2, security group).

---

## 10. More documentation

Inside `genai-doc-assistant-capstone/docs/`:

| Doc | Topic |
|-----|-------|
| `DOCKER.md` | Local Docker details |
| `CONFIGURATION.md` | All env vars |
| `ARCHITECTURE.md` | How agents work |
| `ARCHITECTURE_DIAGRAM.md` | Diagrams |
| `EC2.md` | Cloud deployment |

---

## 11. Checklist for person receiving the folder

- [ ] Folder contains **both** `pycorekit/` and `genai-doc-assistant-capstone/`
- [ ] Docker Desktop installed and running
- [ ] Copied `.env.example` → `.env`
- [ ] Added own `GROQ_API_KEY` to `.env`
- [ ] Ran `docker compose up --build`
- [ ] Opened http://localhost:8501
- [ ] Uploaded a test PDF and asked a question

---

*GenAI Document Assistant — Multi-Agent RAG Capstone*
