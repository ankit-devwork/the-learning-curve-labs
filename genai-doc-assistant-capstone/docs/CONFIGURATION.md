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
