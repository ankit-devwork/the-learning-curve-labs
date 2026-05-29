import os
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import declarative_base
from app.core.load_property import settings
from app.observability.logger import logger, get_current_correlation_id

Base = declarative_base()

class DatabaseService:
    """
    Central service handling async PostgreSQL connection infrastructure.
    Manages connection pooling and provides isolated database sessions per request.
    """
    def __init__(self):
        # 1. Check for an absolute, direct database connection string override first
        env_db_url = os.getenv("DATABASE_URL")
        
        if env_db_url:
            self.db_url = env_db_url
            # Parse components out of the URL for psycopg compatibility later
            # Strips prefix and breaks down user:pass@host:port/db
            try:
                remainder = env_db_url.split("://")[1]
                credentials, host_db = remainder.split("@")
                self.db_user, self.db_password = credentials.split(":")
                host_port, self.db_name = host_db.split("/")
                self.db_host, self.db_port = host_port.split(":")
            except Exception:
                # Fallback defaults if URL parsing fails for helper methods
                self.db_user = os.getenv("DATABASE_USER", getattr(settings, "db_user", "postgres"))
                self.db_password = os.getenv("DATABASE_PASSWORD", getattr(settings, "db_password", "password"))
                self.db_host = "postgres-vector"
                self.db_port = "5432"
                self.db_name = "digital_worker_db"
        else:
            # 2. Fall back to building it piece-by-piece from environment variables or properties
            self.db_user = os.getenv("DATABASE_USER", getattr(settings, "db_user", "postgres"))
            self.db_password = os.getenv("DATABASE_PASSWORD", getattr(settings, "db_password", "password"))
            self.db_host = os.getenv("DATABASE_HOST", getattr(settings, "db_host", "localhost"))
            self.db_port = os.getenv("DATABASE_PORT", getattr(settings, "db_port", "5432"))
            self.db_name = os.getenv("DATABASE_NAME", getattr(settings, "db_name", "digital_worker_db"))
            self.db_url = f"postgresql+asyncpg://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
            
        # 2. Create the Database Engine with a Connection Pool
        self.engine = create_async_engine(
            self.db_url,
            echo=False,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True
        )
        
        # 3. Create a Session Factory
        self.session_factory = async_sessionmaker(
            bind=self.engine,
            autoflush=False,
            expire_on_commit=False,
            class_=AsyncSession
        )

    def get_psycopg_url(self) -> str:
        """
        Assembles a clean connection string compatible with native psycopg drivers.
        Drops the '+asyncpg' dialect prefix since psycopg_pool requires standard URLs.
        """
        # If an absolute override URL is found, strip out the asyncpg driver component
        env_db_url = os.getenv("DATABASE_URL")
        if env_db_url:
            return env_db_url.replace("+asyncpg", "")
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"

    def _get_bound_logger(self):
        """Helper to get a logger stamped with the active user's ContextVar ID"""
        return logger.bind(correlation_id=get_current_correlation_id(), component="Database")

    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Dependency Provider for FastAPI Routes.
        Yields an isolated database session that automatically closes when done.
        """
        log = self._get_bound_logger()
        session = self.session_factory()
        
        try:
            log.debug("Database session opened successfully.")
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            log.error("Database transaction rolled back due to error", error=str(e))
            raise
        finally:
            await session.close()
            log.debug("Database session closed and returned to pool.")

    async def close(self):
        """Disconnect the engine pool when the application shuts down."""
        logger.info("Closing PostgreSQL database connection pool...")
        await self.engine.dispose()

# Create a singleton instance to manage the master pool
db_service = DatabaseService()