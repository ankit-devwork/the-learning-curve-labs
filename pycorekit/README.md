# pycorekit

`pycorekit` is a plug‑and‑play **observability**, **logging**, **caching**, **tracing**, **configuration**, and **exception-handling** toolkit for Python and FastAPI services.

It is designed to be reusable across multiple backend projects with zero friction — a true internal infrastructure framework.

---

# ✨ Features

### 🔹 Logging
- Daily log rotation
- Correlation ID binding
- Worker-safe logging
- JSON-friendly formatting
- Async-safe queueing

### 🔹 Correlation IDs
- Automatic extraction or generation
- ContextVar-based propagation
- FastAPI middleware included

### 🔹 Tracing
- Unified Langfuse + LangSmith tracing
- Span decorators
- Automatic duration measurement

### 🔹 Caching
- Unified Redis client (Upstash REST + local Redis)
- Final-answer caching (GraphRAG / LLM pipelines)
- SHA256 stable cache keys
- TTL support
- Pattern-based invalidation

### 🔹 Configuration
- YAML loader (safe, cached)
- Typed config loader (Pydantic)
- ConfigLoader with env overrides + auto-reload

### 🔹 Exceptions
- Custom exception class
- FastAPI exception handler

---

# 📦 Installation

Local development:

```bash
pip install -e .


📁 Project Structure
pycorekit/
    logging/
    correlation/
    tracing/
    cache/
    exceptions/
    utils/

🚀 Quick Start

1. Add Correlation ID Middleware

from fastapi import FastAPI
from pycorekit.correlation.middleware import CorrelationIdMiddleware

app = FastAPI()
app.add_middleware(CorrelationIdMiddleware)

🧩 Logging
Import logger
from pycorekit.logging.logger import get_logger

log = get_logger("startup")
log.info("Service started")

Worker logger (background tasks)
from pycorekit.logging.worker_logger import get_worker_logger

log = get_worker_logger()
log.info("Background job running")

🧩 Correlation IDs
Get current correlation ID
from pycorekit.correlation.context import get_current_correlation_id

cid = get_current_correlation_id()


🧩 Tracing
Use tracing decorator

from pycorekit.tracing.decorators import with_observability
@with_observability("generate_summary")
async def generate_summary(text: str):
    return "summary..."

🧩 Unified Redis
Works with:

Upstash REST API

Local Redis server

Usage

from pycorekit.cache.unified_redis import redis
pong = await redis.ping()

🧩 Cache Service

from pycorekit.cache.unified_redis import redis
from pycorekit.cache.cache_service import CacheService

cache = CacheService(redis)