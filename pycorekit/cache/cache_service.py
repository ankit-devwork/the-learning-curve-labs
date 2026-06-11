"""
CacheService

A high-level caching abstraction built on top of UnifiedRedis.

Features:
- Final-answer caching (GraphRAG / LLM pipelines)
- Stable SHA256 cache keys
- JSON serialization
- Configurable TTL
- Fully async
- Plug-and-play for any FastAPI or Python backend

This service intentionally keeps the API minimal and predictable.
"""

import json
from typing import Any, Dict, Optional
from .cache_keys import make_query_cache_key


class CacheService:
    """
    High-level cache wrapper that provides:
    - Deterministic final-answer caching
    - Automatic key generation
    - JSON serialization/deserialization
    - TTL support

    This class does NOT know anything about Redis internals.
    It only depends on the UnifiedRedis interface.
    """

    def __init__(self, redis_client):
        """
        Args:
            redis_client: Instance of UnifiedRedis (Upstash or local Redis)
        """
        self.redis = redis_client

    # ---------------------------------------------------------
    # FINAL ANSWER CACHE (GraphRAG / LLM Pipelines)
    # ---------------------------------------------------------

    async def get_final(self, thread_id: str, question: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a cached final answer for a thread/question pair.

        Args:
            thread_id (str): Session or conversation ID.
            question (str): User question text.

        Returns:
            dict or None: Cached payload, or None if not found.

        Example:
            answer = await cache.get_final(thread_id, question)
        """
        key = make_query_cache_key(thread_id, question)
        raw = await self.redis.get(key)
        return json.loads(raw) if raw else None

    async def set_final(
        self,
        thread_id: str,
        question: str,
        payload: Dict[str, Any],
        ttl: int = 86400,
    ) -> None:
        """
        Store a final answer payload in the cache.

        Args:
            thread_id (str): Session or conversation ID.
            question (str): User question text.
            payload (dict): JSON-serializable payload.
            ttl (int): Time-to-live in seconds (default: 24 hours).

        Example:
            await cache.set_final(thread_id, question, {"answer": "..."})
        """
        key = make_query_cache_key(thread_id, question)
        await self.redis.set(key, json.dumps(payload), ttl)

    # ---------------------------------------------------------
    # GENERIC GET/SET (Optional utility)
    # ---------------------------------------------------------

    async def get(self, key: str) -> Optional[Any]:
        """
        Generic GET wrapper for arbitrary keys.
        """
        raw = await self.redis.get(key)
        try:
            return json.loads(raw)
        except Exception:
            return raw

    async def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        """
        Generic SET wrapper for arbitrary keys.
        """
        if not isinstance(value, str):
            value = json.dumps(value)
        await self.redis.set(key, value, ttl)

    # ---------------------------------------------------------
    # CACHE INVALIDATION
    # ---------------------------------------------------------

    async def delete_pattern(self, pattern: str) -> int:
        """
        Delete all keys matching a pattern.

        Example:
            await cache.delete_pattern("final_answer:*")

        Returns:
            int: Number of keys deleted
        """
        return await self.redis.delete_pattern(pattern)
