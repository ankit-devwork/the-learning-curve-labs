"""
High-level cache service for:
- Final answer caching
- TTL management
"""

import json
from .cache_keys import make_query_cache_key

class CacheService:
    def __init__(self, redis_client):
        self.redis = redis_client

    async def get_final(self, thread_id, question):
        key = make_query_cache_key(thread_id, question)
        val = await self.redis.get(key)
        return json.loads(val) if val else None

    async def set_final(self, thread_id, question, payload, ttl=86400):
        key = make_query_cache_key(thread_id, question)
        await self.redis.set(key, json.dumps(payload), ttl)
