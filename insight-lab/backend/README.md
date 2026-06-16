# FastAPI + LangGraph backend for InsightLab

## Python environment (Conda — recommended)

Requires [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda.

### First-time setup

```powershell
cd backend

# Create env from environment.yml (name: insightlab)
conda env create -f environment.yml

# Activate (prompt shows (insightlab) instead of (base))
conda activate insightlab

# Secrets
copy .env.example .env
# Edit .env — Supabase, Neo4j, Redis, GROQ_API_KEY
```

```bash
# macOS / Linux
cd backend
conda env create -f environment.yml
conda activate insightlab
cp .env.example .env
```

### Run API

```powershell
conda activate insightlab
uvicorn app.main:app --reload --port 8000
```

- API: http://localhost:8000  
- Docs: http://localhost:8000/docs  
- Health: http://localhost:8000/health  

### After pulling dependency changes

```powershell
conda activate insightlab
pip install -r requirements.txt
```

Or recreate the env:

```powershell
conda env remove -n insightlab
conda env create -f environment.yml
```

### Optional: pycorekit (observability / cache)

From a sibling clone of [the-learning-curve-labs](https://github.com/ankit-devwork/the-learning-curve-labs):

```powershell
conda activate insightlab
pip install -e ..\..\the-learning-curve-labs\pycorekit
```

### Verify

```powershell
conda activate insightlab
python -c "import fastapi, neo4j, redis, litellm; print('OK')"
python --version
```

---

## Alternative: venv

If you prefer not to use Conda:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness |
| GET | `/ready` | Readiness — pings Redis, Neo4j, Supabase (503 if configured service fails) |
| GET | `/docs` | OpenAPI UI |

Feature routes (Excel, documents, quiz) will be added in Phase 1.
