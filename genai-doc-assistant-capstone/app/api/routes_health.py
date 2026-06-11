"""
Health and readiness endpoints.
"""

import os
import asyncio

from fastapi import APIRouter
from pycorekit.core_logging.logger import get_logger
from app.service.db_connection import async_get_collection, get_embedding_model

router = APIRouter(tags=["Health"])
log = get_logger("health")


@router.get("/health", summary="Liveness check", operation_id="healthCheck")
async def health_check():
    log.info("Liveness check invoked")
    return {"status": "ok", "message": "API is running"}


@router.get("/ready", summary="Readiness check", operation_id="readinessCheck")
async def readiness_check():
    checks = {
        "api": "ok",
        "chroma": "unknown",
        "embedding_model": "unknown",
        "llm_api_key": "unknown",
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

    status = "ok" if all(
        v == "ok" or v == "configured" for v in checks.values()
    ) else "degraded"

    return {"status": status, "checks": checks}
