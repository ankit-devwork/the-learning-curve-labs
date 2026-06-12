# Configuration

## Files

| File | Purpose |
|------|---------|
| `config.yaml` | Base application settings |
| `.env` | Secrets and runtime overrides |

## Load order

1. `config.yaml`
2. `.env` (via `python-dotenv`)
3. Environment variables with `APP_` prefix

## Override examples

```bash
APP_ENV=prod
APP_RAG__CHUNK_SIZE=500
APP_MODELS__LLM_MODEL=groq/llama-3.3-70b-versatile
APP_FILE_UPLOAD__MAX_FILE_SIZE_MB=25
APP_CACHE__ENABLED=true
APP_CACHE__TTL_SECONDS=7200
```

Nested keys in YAML map to env vars using `__` (double underscore).

## Secrets

API keys are not stored in `config.yaml`. Set them in `.env`:

```bash
GROQ_API_KEY=...
OPENAI_API_KEY=...
```

## Redis (optional)

```bash
REDIS_HOST=localhost
REDIS_PORT=6379
QUERY_CACHE_TTL=3600
```

When Redis is not configured, the capstone app uses an in-memory cache.

## External observability (optional)

Set these in `.env` to export traces to Langfuse and/or LangSmith. In-process spans always work; external tools add a hosted UI.

### Langfuse

```bash
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

### LangSmith

```bash
LANGCHAIN_API_KEY=lsv2_...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=genai-doc-assistant
```

Check configuration at `GET /ready` under `external_tracing`, or in the Streamlit **RAG Observability Dashboard** under **External Tracing**.

## Deployment files

Configuration and secrets are loaded the same way in all environments. Deployment manifests live in the capstone folder:

| File | Environment |
|------|-------------|
| `docker-compose.yml` | Local Docker |
| `docker-compose.dev.yml` | Local Docker (dev) |
| `docker-compose.ecr.yml` | EC2 / ECR (one repo, two tags) |
| `.env.ecr.example` | `ECR_REGISTRY`, `ECR_REPOSITORY`, `BACKEND_IMAGE_TAG`, `STREAMLIT_IMAGE_TAG` |

See [DOCKER.md](DOCKER.md) and [EC2.md](EC2.md) for the verified EC2 runbook.
