"""
Query answer cache for the capstone backend.

Uses pycorekit MemoryCache by default. Set REDIS_HOST to switch to Redis.
"""

import os

from pycorekit.cache.cache_service import CacheService
from pycorekit.cache.memory_cache import MemoryCache
from pycorekit.cache.unified_redis import UnifiedRedis
from app.core.settings import settings

_BACKEND = None
_SERVICE = None


def _build_backend():
    redis_host = os.getenv("REDIS_HOST", "").strip()
    if redis_host:
        return UnifiedRedis()
    return MemoryCache()


def get_query_cache() -> CacheService | None:
    global _BACKEND, _SERVICE
    if not settings.cache.enabled:
        return None
    if _SERVICE is None:
        _BACKEND = _build_backend()
        _SERVICE = CacheService(_BACKEND)
    return _SERVICE


async def get_cached_answer(thread_id: str, question: str) -> dict | None:
    cache = get_query_cache()
    if cache is None:
        return None
    return await cache.get_final(thread_id, question)


async def set_cached_answer(thread_id: str, question: str, payload: dict) -> None:
    cache = get_query_cache()
    if cache is None:
        return
    await cache.set_final(
        thread_id,
        question,
        payload,
        ttl=settings.cache.ttl_seconds,
    )


async def invalidate_query_cache() -> int:
    cache = get_query_cache()
    if cache is None:
        return 0
    deleted = await cache.delete_pattern("final_answer:*")
    if _BACKEND is not None and hasattr(_BACKEND, "clear"):
        await _BACKEND.clear()
    return deleted
