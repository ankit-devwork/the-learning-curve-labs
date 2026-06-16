import time
from typing import Any

from neo4j import AsyncDriver, AsyncGraphDatabase
from neo4j.exceptions import Neo4jError, ServiceUnavailable

from app.core.config import settings


class Neo4jClient:
    def __init__(self) -> None:
        self._driver: AsyncDriver | None = None

    @property
    def is_configured(self) -> bool:
        return bool(settings.neo4j_uri and settings.neo4j_user and settings.neo4j_password)

    def get_driver(self) -> AsyncDriver:
        if self._driver is None:
            self._driver = AsyncGraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_user, settings.neo4j_password),
            )
        return self._driver

    async def ping(self) -> dict[str, Any]:
        if not self.is_configured:
            return {"ok": False, "error": "not configured"}

        start = time.perf_counter()
        try:
            driver = self.get_driver()
            async with driver.session() as session:
                result = await session.run("RETURN 1 AS ping")
                record = await result.single()
                if record and record["ping"] == 1:
                    return {
                        "ok": True,
                        "latency_ms": round((time.perf_counter() - start) * 1000, 2),
                    }
                return {"ok": False, "error": "unexpected response"}
        except ServiceUnavailable as exc:
            return {"ok": False, "error": f"service unavailable: {exc}"}
        except Neo4jError as exc:
            return {"ok": False, "error": f"neo4j error: {exc.message}"}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    async def close(self) -> None:
        if self._driver is not None:
            await self._driver.close()
            self._driver = None


neo4j_client = Neo4jClient()
