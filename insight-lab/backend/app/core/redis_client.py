import os
from typing import Optional

import httpx
import redis.asyncio as aioredis

from app.core.config import settings


class RedisClient:
    """Local Redis or Upstash REST — used for cache and readiness checks."""

    def __init__(self) -> None:
        self.use_upstash = bool(settings.upstash_redis_rest_url and settings.upstash_redis_rest_token)
        self._client: aioredis.Redis | None = None
        self._http: httpx.AsyncClient | None = None

    @property
    def is_configured(self) -> bool:
        return self.use_upstash or bool(settings.redis_host)

    async def ping(self) -> bool:
        if not self.is_configured:
            return False

        if self.use_upstash:
            if self._http is None:
                self._http = httpx.AsyncClient(timeout=5.0)
            resp = await self._http.post(
                settings.upstash_redis_rest_url,
                headers={"Authorization": f"Bearer {settings.upstash_redis_rest_token}"},
                json=["PING"],
            )
            resp.raise_for_status()
            return resp.json().get("result") in ("PONG", "OK", None)

        if self._client is None:
            self._client = aioredis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                db=settings.redis_db,
                password=settings.redis_password or None,
                decode_responses=True,
                socket_connect_timeout=5,
            )
        return bool(await self._client.ping())

    @property
    def mode(self) -> str:
        if not self.is_configured:
            return "not_configured"
        return "upstash" if self.use_upstash else "local"

    async def close(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None
        if self._client is not None:
            await self._client.close()
            self._client = None


redis_client = RedisClient()
