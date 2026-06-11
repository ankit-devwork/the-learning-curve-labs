# pycorekit

`pycorekit` is a plug-and-play toolkit for Python backends and FastAPI services. It provides:

- Observability
- Logging
- Tracing
- Correlation ID propagation
- Caching
- Configuration loading
- Exception handling

The package is designed to be reusable across multiple projects with minimal setup.

---

# ✨ Features

## 🔹 Logging
- Structured log binding with module and correlation ID
- Daily log rotation
- Optional JSON logs
- Async-safe output

## 🔹 Correlation Context
- `ContextVar`-based correlation ID propagation
- Simple API for request-scoped IDs

## 🔹 Tracing
- Built-in Langfuse + LangSmith tracing support
- FastAPI route decorator for observability
- Automatic duration measurement and span lifecycle management

## 🔹 Caching
- Final-answer cache helper
- Stable SHA256 query key generation
- JSON-safe Redis get/set
- TTL support and pattern delete

## 🔹 Configuration
- Safe YAML loading
- Typed config validation
- Base directory injection for path config

## 🔹 Exceptions
- Structured `AppException`
- Domain-specific `FileException`
- FastAPI exception handlers for consistent responses

---

# 📦 Installation

Install locally for development:

```bash
pip install -e .
```

---

# 📁 Package Layout

```text
pycorekit/
  cache/
  correlation/
  exceptions/
  logging/
  observability/
  tracing/
  utils/
```

---

# 🚀 Quick Start

## 1. Initialize logging

```python
from pycorekit.logging.logger import init_logger, get_logger

init_logger(
    log_dir="logs",
    rotation="00:00",
    retention="7 days",
    json_file=True,
)

log = get_logger("startup")
log.info("Service started")
```

## 2. Use correlation IDs

```python
from pycorekit.correlation.context import (
    generate_correlation_id,
    set_current_correlation_id,
    get_current_correlation_id,
)

cid = generate_correlation_id()
set_current_correlation_id(cid)
log.info("correlation set", correlation_id=cid)
```

## 3. Add request tracing middleware

```python
from fastapi import FastAPI
from pycorekit.tracing.middleware import RequestTracingMiddleware

app = FastAPI()
app.add_middleware(RequestTracingMiddleware)
```

This middleware initializes the request trace and adds `x-correlation-id` to responses.

## 4. Decorate FastAPI routes for observability

```python
from fastapi import APIRouter, Request
from pycorekit.tracing.decorators import with_observability

router = APIRouter()

@router.post("/upload")
@with_observability("upload_and_ingest")
async def upload(request: Request):
    return {"status": "ok"}
```

When the decorated route returns a dict, `pycorekit` automatically injects a sanitized `observability` payload.

## 5. Sanitize observability traces

```python
from pycorekit.utils.sanitize_observability import sanitize_observability

safe_trace = sanitize_observability(request.state.trace)
```

Use this helper to convert raw trace objects into frontend-safe JSON.

---

# 🧰 Utilities

## File uploader

```python
from pycorekit.utils.uploader import upload_bytes

saved_path = await upload_bytes(
    data=file_bytes,
    dest="local",
    dest_dir="./uploads",
    dest_name="document.pdf",
)
```

Supports both local filesystem storage and S3 upload.

## YAML configuration loader

```python
from pathlib import Path
from pycorekit.utils.config_loader import ConfigLoader

config = ConfigLoader(
    Path("config.yaml"),
    base_dir=Path("."),
    env_file=Path(".env"),
    env_prefix="APP",
)
raw = config.load()
```

For typed model validation:

```python
settings = config.load_typed(Settings)
```

### Dynamic `.env` and environment overrides

`ConfigLoader` supports runtime overrides without editing YAML:

1. **`.env` file** — loaded automatically when `env_file` is set. Secrets like `GROQ_API_KEY` are placed in `os.environ` for the rest of the app.
2. **Config overrides** — env vars override YAML using nested keys with `__`:

| `.env` variable | Overrides `config.yaml` key |
|-----------------|----------------------------|
| `APP_ENV=prod` | `env` (with `env_prefix="APP"`) |
| `APP_RAG__CHUNK_SIZE=500` | `rag.chunk_size` |
| `APP_MODELS__LLM_MODEL=groq/...` | `models.llm_model` |
| `APP_FILE_UPLOAD__MAX_FILE_SIZE_MB=10` | `file_upload.max_file_size_mb` |

Without a prefix, use keys like `RAG__CHUNK_SIZE=500` directly.

Type coercion follows the existing YAML value (int, float, bool, list).

---

# 🗃 Caching

```python
from pycorekit.cache.cache_service import CacheService
from pycorekit.cache.unified_redis import redis

cache = CacheService(redis)
await cache.set_final(thread_id, question, payload)
result = await cache.get_final(thread_id, question)
```

Cache keys are normalized and stable using SHA256 hashing.

---

# 🚨 Exception handling

```python
from pycorekit.exceptions.base import AppException
from pycorekit.exceptions.handlers import (
    app_exception_handler,
    generic_exception_handler,
)

app.add_exception_handler(AppException, app_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)
```

Use `AppException` for known application errors and `FileException` for file-specific failures.

---

# 📘 Notes

- `pycorekit` is intentionally minimal and lightweight.
- The tracing layer is optimized for production observability with Langfuse and LangSmith.
- The package is suitable for backend services, FastAPI APIs, and async workflows.
