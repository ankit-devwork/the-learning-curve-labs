# app/core/graph_db.py
from neo4j import AsyncGraphDatabase, AsyncDriver
from app.core.load_property import settings
from app.observability.logger import logger

class Neo4jService:
    def __init__(self):
        self.uri = settings.neo4j_uri
        self.user = settings.neo4j_user
        self.password = settings.neo4j_password
        self._driver: AsyncDriver | None = None

    def get_driver(self) -> AsyncDriver:
        """Returns the shared asynchronous connection pool driver."""
        if self._driver is None:
            self._driver = AsyncGraphDatabase.driver(
                self.uri, 
                auth=(self.user, self.password)
            )
        return self._driver

    async def close(self):
        """Gracefully terminates open sockets back to the local database cluster."""
        if self._driver:
            logger.info("Closing Neo4j Graph Database connection pool...")
            await self._driver.close()

graph_service = Neo4jService()