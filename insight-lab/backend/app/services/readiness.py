import time
from typing import Any

import httpx

from app.core.config import settings
from app.core.neo4j_client import neo4j_client
from app.core.redis_client import redis_client


async def check_redis() -> dict[str, Any]:
    if not redis_client.is_configured:
        return {"status": "not_configured", "mode": "not_configured"}

    start = time.perf_counter()
    try:
        ok = await redis_client.ping()
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        if ok:
            return {"status": "ok", "mode": redis_client.mode, "latency_ms": latency_ms}
        return {"status": "error", "mode": redis_client.mode, "error": "ping failed"}
    except Exception as exc:
        return {
            "status": "error",
            "mode": redis_client.mode,
            "error": str(exc),
        }


async def check_neo4j() -> dict[str, Any]:
    if not neo4j_client.is_configured:
        return {"status": "not_configured"}

    result = await neo4j_client.ping()
    if result.get("ok"):
        return {
            "status": "ok",
            "latency_ms": result.get("latency_ms"),
        }
    return {"status": "error", "error": result.get("error", "unknown error")}


async def check_supabase() -> dict[str, Any]:
    url = settings.supabase_url.strip()
    if not url:
        return {"status": "not_configured"}

    if not settings.supabase_service_role_key.strip() or not settings.supabase_jwt_secret.strip():
        return {
            "status": "error",
            "error": "SUPABASE_URL set but service_role or JWT secret missing",
        }

    health_url = f"{url.rstrip('/')}/auth/v1/health"
    start = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(health_url)
            resp.raise_for_status()
            body = resp.json()
            latency_ms = round((time.perf_counter() - start) * 1000, 2)
            if body.get("healthy") is True or resp.status_code == 200:
                return {"status": "ok", "latency_ms": latency_ms}
            return {"status": "error", "error": f"unexpected health response: {body}"}
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


def check_llm() -> dict[str, Any]:
    if settings.groq_api_key.strip():
        return {"status": "configured", "provider": "groq"}
    return {"status": "not_configured"}


def aggregate_status(checks: dict[str, dict[str, Any]]) -> tuple[str, int]:
    """
    ready       — all configured dependencies are ok
    degraded    — at least one configured dependency failed
    """
    configured_checks = [
        c for c in checks.values() if c.get("status") not in ("not_configured", None)
    ]
    if not configured_checks:
        return "ready", 200

    failures = [c for c in configured_checks if c.get("status") not in ("ok", "configured")]
    if failures:
        return "degraded", 503
    return "ready", 200


async def run_readiness_checks() -> tuple[dict[str, Any], int]:
    checks = {
        "api": {"status": "ok"},
        "redis": await check_redis(),
        "neo4j": await check_neo4j(),
        "supabase": await check_supabase(),
        "llm": check_llm(),
    }
    overall, status_code = aggregate_status(checks)
    return {"status": overall, "checks": checks}, status_code
