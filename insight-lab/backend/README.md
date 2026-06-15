# FastAPI + LangGraph backend for InsightLab

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --port 8000
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness |
| GET | `/ready` | Readiness (checks configured services) |
| GET | `/docs` | OpenAPI UI |

Feature routes (Excel, documents, quiz) will be added in Phase 1.
