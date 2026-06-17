import json
import time
from typing import Any

from pycorekit.cache import CacheService
from pycorekit.cache.unified_redis import UnifiedRedis

from app.core.yaml_config import get_yaml_config

_unified_redis = UnifiedRedis()
cache_service = CacheService(_unified_redis)


def _prefix(key: str) -> str:
    prefix = get_yaml_config().cache.key_prefix
    return f"{prefix}:{key}"


async def cache_get(key: str) -> Any | None:
    return await cache_service.get(_prefix(key))


async def cache_set(key: str, value: Any, ttl: int) -> None:
    await cache_service.set(_prefix(key), value, ttl)


async def cache_delete(key: str) -> None:
    raw = await _unified_redis.get(_prefix(key))
    if raw is not None:
        await _unified_redis.delete_pattern(_prefix(key))


async def check_rate_limit(*, key: str, limit: int, window_seconds: int) -> tuple[bool, int]:
    """
    Fixed-window rate limit using Redis get/set.
    Returns (allowed, retry_after_seconds).
    """
    full_key = _prefix(f"ratelimit:{key}")
    now = int(time.time())
    window_start = now - (now % window_seconds)

    raw = await _unified_redis.get(full_key)
    if raw:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {"count": 0, "window_start": window_start}
    else:
        payload = {"count": 0, "window_start": window_start}

    if payload.get("window_start") != window_start:
        payload = {"count": 0, "window_start": window_start}

    payload["count"] = int(payload.get("count", 0)) + 1
    await _unified_redis.set(full_key, json.dumps(payload), window_seconds)

    if payload["count"] > limit:
        retry_after = window_seconds - (now - window_start)
        return False, max(retry_after, 1)
    return True, 0


async def close_cache() -> None:
    await _unified_redis.close()
