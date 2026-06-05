import os
import json
import httpx
import redis.asyncio as aioredis
from typing import Optional, List
from app.core.load_property import settings
from app.observability.logger import logger, get_current_correlation_id

class UnifiedRedis:
    def __init__(self):
        self.use_upstash = bool(
            os.getenv("UPSTASH_REDIS_REST_URL") and os.getenv("UPSTASH_REDIS_REST_TOKEN")
        )

        if self.use_upstash:
            self.upstash_url = os.getenv("UPSTASH_REDIS_REST_URL")
            self.upstash_token = os.getenv("UPSTASH_REDIS_REST_TOKEN")
            self.client = httpx.AsyncClient()
        else:
            self.client = aioredis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                db=settings.redis_db,
                password=settings.redis_password,
                decode_responses=True,
            )

    # -------------------------
    # PING
    # -------------------------
    async def ping(self):
        if self.use_upstash:
            resp = await self.client.post(
                self.upstash_url,
                headers={"Authorization": f"Bearer {self.upstash_token}"},
                json=["PING"],
            )
            return resp.json().get("result") in ("PONG", "OK", None)
        return await self.client.ping()

    # -------------------------
    # GET
    # -------------------------
    async def get(self, key: str):
        if self.use_upstash:
            resp = await self.client.post(
                self.upstash_url,
                headers={"Authorization": f"Bearer {self.upstash_token}"},
                json=["GET", key],   # <-- FIXED FORMAT
            )
            return resp.json().get("result")
        return await self.client.get(key)

    # -------------------------
    # SET
    # -------------------------
    async def set(self, key: str, value: str, ttl: int = None):
        ttl = ttl or settings.redis_default_ttl

        if self.use_upstash:
            await self.client.post(
                self.upstash_url,
                headers={"Authorization": f"Bearer {self.upstash_token}"},
                json=["SETEX", key, ttl, value],   # <-- FIXED FORMAT
            )
            return True

        return await self.client.set(key, value, ex=ttl)

    # -------------------------
    # DELETE PATTERN
    # -------------------------
    async def delete_pattern(self, pattern: str):
        if self.use_upstash:
            resp = await self.client.post(
                self.upstash_url,
                headers={"Authorization": f"Bearer {self.upstash_token}"},
                json=["KEYS", pattern],
            )
            keys = resp.json().get("result") or []
            if keys:
                await self.client.post(
                    self.upstash_url,
                    headers={"Authorization": f"Bearer {self.upstash_token}"},
                    json=["DEL"] + keys,
                )
            return len(keys)

        keys = await self.client.keys(pattern)
        if keys:
            await self.client.delete(*keys)
        return len(keys)




# Singleton instance
redis = UnifiedRedis()
