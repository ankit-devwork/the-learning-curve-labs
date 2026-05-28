import json
import logging
import redis.asyncio as aioredis
from typing import Optional, List
from app.core.load_property import settings

logger = logging.getLogger("app.services.cache_service")

class CacheService:
    """
    Asynchronous Redis Caching Service tailored for the Digital Worker Studio.
    
    Handles standard Absolute TTL caching, session-aware Sliding Response TTL
    caching, and automated query namespace cache invalidation upon document ingestion.
    """
    
    def __init__(self):
        """Initializes the service shell wrapper with an unassigned Redis connection engine."""
        self.redis: Optional[aioredis.Redis] = None

    def initialize(self):
        """
        Instantiates the asynchronous Redis connection client.
        This must be called during the FastAPI lifecycle startup sequence.
        """
        self.redis = aioredis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            password=settings.redis_password,
            decode_responses=True  # Coerces incoming byte strings directly into Python UTF-8 strings
        )
        logger.info("Async Redis Cache Engine successfully initialized.")

    # -----------------------------------------------------------------------------------------
    # STRATEGY A: SLIDING RESPONSE TTL (Session/Traffic-Aware Caching)
    # -----------------------------------------------------------------------------------------
    async def get_with_sliding_ttl(
        self, 
        query: str, 
        extend_seconds: int = getattr(settings, "sliding_response_ttl", 1800)
    ) -> Optional[dict]:
        """
        Retrieves a cached item from Redis and extends its expiration window upon hit.
        
        Args:
            query (str): The raw text question submitted by the user.
            extend_seconds (int): Time in seconds to reset the key expiration. 
                                  Defaults to settings configuration or a fall-through of 30 mins.
        Returns:
            Optional[dict]: The parsed JSON response payload, or None if a cache miss occurs.
        """
        if not self.redis:
            logger.warning("Cache retrieval bypassed: Redis client connection pool is inactive.")
            return None
            
        try:
            cache_key = f"query:{query.strip().lower()}"
            
            # 1. Inspect the cache container for an existing key hit
            cached_data = await self.redis.get(cache_key)
            
            if cached_data:
                # 2. Only push an updated expiration flag if the key actually exists in memory.
                # This guarantees we don't cause ghost-key creations on multi-node workers.
                await self.redis.expire(cache_key, extend_seconds)
                logger.info(f"[Sliding Hit] Extended Response TTL window by {extend_seconds}s for key: {cache_key}")
                return json.loads(cached_data)
                
            logger.info(f"[Cache Miss] No records found matching query key: {cache_key}")
            
        except Exception as e:
            logger.error(f"Failed to execute Sliding Response TTL search sequence: {e}")
            
        return None

    # -----------------------------------------------------------------------------------------
    # STRATEGY B: FIXED TTL (Absolute Countdown Caching)
    # -----------------------------------------------------------------------------------------
    async def set_fixed_ttl(
        self, 
        query: str, 
        response_payload: dict, 
        expire_seconds: int = getattr(settings, "redis_default_ttl", 3600)
    ):
        """
        Commits a generated pipeline response payload to Redis using a strict absolute countdown.
        
        Args:
            query (str): The raw user query serving as the descriptive routing key identifier.
            response_payload (dict): The compiled dictionary payload from the LangGraph canvas.
            expire_seconds (int): Lifespan window before automatic database structural eviction.
                                  Defaults to settings layout configuration or 1 hour fall-through.
        """
        if not self.redis:
            return
            
        try:
            cache_key = f"query:{query.strip().lower()}"
            
            # Execute atomic SET with EXpiration (SETEX) routine
            await self.redis.setex(
                cache_key,
                expire_seconds,
                json.dumps(response_payload)
            )
            logger.info(f"[Fixed Store] Saved execution payload under key '{cache_key}' with strict TTL of {expire_seconds}s.")
            
        except Exception as e:
            logger.error(f"Failed to commit absolute fixed cache record block: {e}")

    # -----------------------------------------------------------------------------------------
    # STRATEGY C: CACHE INVALIDATION (Real-Time Synchronicity Control)
    # -----------------------------------------------------------------------------------------
    async def flush_all_query_caches(self):
        """
        Scans the database and drops all keys under the 'query:*' namespace layout.
        
        Crucial for multi-source ingestion loops. This must be triggered when a new 
        PDF document is successfully saved to prevent stale retrieval outputs.
        """
        if not self.redis:
            return
            
        try:
            logger.info("New document ingestion detected. Launching full system cache invalidation scan...")
            
            # Collect all keys stored under the query namespace
            query_keys: List[str] = await self.redis.keys("query:*")
            
            if query_keys:
                # Unpack and wipe the keys in a single atomic deletion routine
                await self.redis.delete(*query_keys)
                logger.info(f"[Cache Invalidation] Successfully purged {len(query_keys)} stale query keys from Redis memory.")
            else:
                logger.info("[Cache Invalidation] Cache namespace was already completely pristine. No elements cleared.")
                
        except Exception as e:
            logger.error(f"Critical breakdown encountered during pipeline cache flush operation: {e}")

# Instantiate the service as a singleton to preserve the connection framework across endpoints
cache_service = CacheService()