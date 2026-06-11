"""
In-memory async cache backend compatible with CacheService and UnifiedRedis.

Used when Redis is not configured (local dev, capstone default).
Supports TTL and simple glob-style pattern deletion.
"""

from __future__ import annotations

import fnmatch
import time
from typing import Any, Dict, Optional, Tuple


class MemoryCache:
    """Async-friendly in-memory key-value store with optional TTL."""

    def __init__(self):
        self._store: Dict[str, Tuple[str, Optional[float]]] = {}

    def _purge_expired(self, key: str) -> None:
        if key not in self._store:
            return
        _, expires_at = self._store[key]
        if expires_at is not None and time.time() > expires_at:
            del self._store[key]

    async def ping(self) -> bool:
        return True

    async def get(self, key: str) -> Optional[str]:
        self._purge_expired(key)
        entry = self._store.get(key)
        return entry[0] if entry else None

    async def set(self, key: str, value: str, ttl: int | None = None) -> bool:
        expires_at = time.time() + ttl if ttl else None
        self._store[key] = (value, expires_at)
        return True

    async def delete_pattern(self, pattern: str) -> int:
        keys = list(self._store.keys())
        matched = [k for k in keys if fnmatch.fnmatch(k, pattern)]
        for key in matched:
            self._store.pop(key, None)
        return len(matched)

    async def clear(self) -> int:
        count = len(self._store)
        self._store.clear()
        return count
