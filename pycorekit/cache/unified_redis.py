"""
Unified Redis client supporting local Redis and Upstash REST API.
"""

import json
import os
from typing import Optional

import httpx
import redis.asyncio as aioredis

from pycorekit.core_logging.logger import get_logger

logger = get_logger(__name__)


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
            self.redis_host = os.getenv("REDIS_HOST", "localhost")
            self.redis_port = int(os.getenv("REDIS_PORT", "6379"))
            self.redis_db = int(os.getenv("REDIS_DB", "0"))
            self.redis_password = os.getenv("REDIS_PASSWORD") or None
            self.client = aioredis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                db=self.redis_db,
                password=self.redis_password,
                decode_responses=True,
            )

    async def ping(self) -> bool:
        if self.use_upstash:
            resp = await self.client.post(
                self.upstash_url,
                headers={"Authorization": f"Bearer {self.upstash_token}"},
                json=["PING"],
            )
            return resp.json().get("result") in ("PONG", "OK", None)
        return bool(await self.client.ping())

    async def get(self, key: str) -> Optional[str]:
        if self.use_upstash:
            resp = await self.client.post(
                self.upstash_url,
                headers={"Authorization": f"Bearer {self.upstash_token}"},
                json=["GET", key],
            )
            return resp.json().get("result")
        return await self.client.get(key)

    async def set(self, key: str, value: str, ttl: int | None = None) -> bool:
        ttl = ttl or int(os.getenv("REDIS_DEFAULT_TTL", "3600"))
        if self.use_upstash:
            await self.client.post(
                self.upstash_url,
                headers={"Authorization": f"Bearer {self.upstash_token}"},
                json=["SETEX", key, ttl, value],
            )
            return True
        return bool(await self.client.set(key, value, ex=ttl))

    async def delete_pattern(self, pattern: str) -> int:
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
                    json=["DEL", *keys],
                )
            return len(keys)

        keys = await self.client.keys(pattern)
        if keys:
            await self.client.delete(*keys)
        return len(keys)

    async def close(self) -> None:
        if self.use_upstash:
            await self.client.aclose()
        else:
            await self.client.close()


redis = UnifiedRedis()
