"""
Health and readiness endpoints.
"""

import os
import asyncio

from fastapi import APIRouter
from pycorekit.core_logging.logger import get_logger
from pycorekit.tracing.tracing import get_external_tracing_status
from app.service.db_connection import async_get_collection, get_embedding_model

router = APIRouter(tags=["Health"])
log = get_logger("health")


@router.get("/health", summary="Liveness check", operation_id="healthCheck")
async def health_check():
    log.info("Liveness check invoked")
    return {"status": "ok", "message": "API is running"}


@router.get("/ready", summary="Readiness check", operation_id="readinessCheck")
async def readiness_check():
    tracing = get_external_tracing_status()
    checks = {
        "api": "ok",
        "chroma": "unknown",
        "embedding_model": "unknown",
        "llm_api_key": "unknown",
        "langfuse": "configured" if tracing["langfuse"]["configured"] else "not_configured",
        "langsmith": "configured" if tracing["langsmith"]["configured"] else "not_configured",
    }

    try:
        collection = await async_get_collection("documents")
        await asyncio.to_thread(collection.count)
        checks["chroma"] = "ok"
    except Exception as exc:
        log.error(f"Chroma readiness failed: {exc}")
        checks["chroma"] = f"error: {exc}"

    try:
        await asyncio.to_thread(get_embedding_model)
        checks["embedding_model"] = "ok"
    except Exception as exc:
        log.error(f"Embedding model readiness failed: {exc}")
        checks["embedding_model"] = f"error: {exc}"

    if os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY"):
        checks["llm_api_key"] = "configured"
    else:
        checks["llm_api_key"] = "missing"

    core_checks = ("api", "chroma", "embedding_model", "llm_api_key")
    status = "ok" if all(
        checks[k] == "ok" or checks[k] == "configured" for k in core_checks
    ) else "degraded"

    return {"status": status, "checks": checks, "external_tracing": tracing}
