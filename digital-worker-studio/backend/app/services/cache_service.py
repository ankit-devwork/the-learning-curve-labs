import json
from app.core.cache_keys import make_query_cache_key
from app.core.unified_redis import redis


class CacheService:
    def __init__(self, redis_client):
        self.redis = redis_client

    # -------------------------------
    # FINAL ANSWER CACHE
    # -------------------------------
    async def get_final_answer(self, thread_id: str, question: str):
        key = make_query_cache_key(thread_id, question)
        value = await self.redis.get(key)
        if value is None:
            return None
        return json.loads(value)

    async def set_final_answer(self, thread_id: str, question: str, answer: dict, ttl: int = 86400):
        key = make_query_cache_key(thread_id, question)
        await self.redis.set(key, json.dumps(answer), ttl)

    # -------------------------------
    # LEGACY SLIDING TTL CACHE
    # -------------------------------
    async def get_with_sliding_ttl(self, query: str, extend_seconds: int = 1800):
        key = f"query:{query.lower()}"
        val = await self.redis.get(key)
        if val:
            await self.redis.set(key, val, extend_seconds)
            return json.loads(val)
        return None

    async def set_fixed_ttl(self, query: str, payload: dict, ttl: int = 3600):
        key = f"query:{query.lower()}"
        await self.redis.set(key, json.dumps(payload), ttl)

    async def flush_all_query_caches(self):
        await self.redis.delete_pattern("query:*")


cache_service = CacheService(redis)
