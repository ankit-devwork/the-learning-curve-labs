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


async def _incr(key: str, ttl: int) -> int:
    full_key = _prefix(key)
    if _unified_redis.use_upstash:
        response = await _unified_redis.client.post(
            _unified_redis.upstash_url,
            headers={"Authorization": f"Bearer {_unified_redis.upstash_token}"},
            json=["INCR", full_key],
        )
        count = int(response.json().get("result") or 0)
        if count == 1:
            await _unified_redis.client.post(
                _unified_redis.upstash_url,
                headers={"Authorization": f"Bearer {_unified_redis.upstash_token}"},
                json=["EXPIRE", full_key, ttl],
            )
        return count

    count = int(await _unified_redis.client.incr(full_key))
    if count == 1:
        await _unified_redis.client.expire(full_key, ttl)
    return count


async def check_rate_limit(*, key: str, limit: int, window_seconds: int) -> tuple[bool, int]:
    """
    Fixed-window rate limit using atomic Redis INCR.
    Returns (allowed, retry_after_seconds).
    """
    now = int(time.time())
    window_start = now - (now % window_seconds)
    window_key = f"ratelimit:{key}:{window_start}"
    count = await _incr(window_key, window_seconds)

    if count > limit:
        retry_after = window_seconds - (now - window_start)
        return False, max(retry_after, 1)
    return True, 0


async def close_cache() -> None:
    await _unified_redis.close()
