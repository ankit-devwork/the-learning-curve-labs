import redis.asyncio as aioredis
from app.core.load_property import settings
from app.observability.logger import logger, get_current_correlation_id
from typing import Optional

class RedisService:
    """
    Central service handling async Redis cache infrastructure.
    Uses connection pooling to handle high multi-user traffic.
    """
    def __init__(self):
        self.pool = aioredis.ConnectionPool(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            password=settings.redis_password,
            decode_responses=True, # Automatically converts bytes to Python strings
            max_connections=20     # Safe cap for concurrent user connection reuse
        )
        self.client = aioredis.Redis(connection_pool=self.pool)

    def _get_bound_logger(self):
        """Helper to get a logger stamped with the active user's ContextVar ID"""
        return logger.bind(correlation_id=get_current_correlation_id(), component="Redis")

    async def get(self, key: str) -> str | None:
        """Fetch data from cache."""
        log = self._get_bound_logger()
        try:
            val = await self.client.get(key)
            if val:
                log.debug(f"Cache HIT for key: {key}")
            else:
                log.debug(f"Cache MISS for key: {key}")
            return val
        except Exception as e:
            log.error(f"Redis GET failed for key {key}", error=str(e))
            return None

       

    async def set(self, key: str, value: str, expire_seconds: Optional[int] = None) -> bool:
        """
        Save data to cache. 
        If expire_seconds is not provided, it falls back to the properties file default.
        """
        log = self._get_bound_logger()
        
        # Fallback to config file if a specific time isn't passed during the call
        ttl = expire_seconds if expire_seconds is not None else settings.redis_default_ttl
        
        try:
            await self.client.set(key, value, ex=ttl)
            log.debug(f"Cache SET successful for key: {key} (TTL: {ttl}s)")
            return True
        except Exception as e:
            log.error(f"Redis SET failed for key {key}", error=str(e))
            return False

    async def close(self):
        """Disconnect the pool when the application shuts down."""
        logger.info("Closing Redis connection pool...")
        await self.pool.disconnect()

# Singleton instance to be shared across all API routes
redis_service = RedisService()