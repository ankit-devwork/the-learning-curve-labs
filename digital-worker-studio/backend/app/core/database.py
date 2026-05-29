import os
from typing import AsyncGenerator
from urllib.parse import urlparse
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
        # 1. Fall back to building it piece-by-piece using your Settings attributes directly.
        # This prioritizes your local properties file as the source of truth for local dev.
        self.db_user = os.getenv("DB_USER", getattr(settings, "db_user", "postgres"))
        self.db_password = os.getenv("DB_PASSWORD", getattr(settings, "db_password", "mypassword"))
        self.db_host = os.getenv("DB_HOST", getattr(settings, "db_host", "localhost"))
        self.db_port = os.getenv("DB_PORT", getattr(settings, "db_port", "5432"))
        self.db_name = os.getenv("DB_NAME", getattr(settings, "db_name", "digital_worker_db"))
        
        # 2. Check for an absolute connection override string, but ONLY use it if we aren't running locally
        env_db_url = os.getenv("DATABASE_URL")
        if env_db_url and not any(local_keyword in env_db_url for local_keyword in ["localhost", "127.0.0.1", "5442"]):
            raw_url = env_db_url.strip("'\"")
            if raw_url.startswith("postgresql://"):
                raw_url = raw_url.replace("postgresql://", "postgresql+asyncpg://", 1)
            elif raw_url.startswith("postgres://"):
                raw_url = raw_url.replace("postgres://", "postgresql+asyncpg://", 1)
                
            self.db_url = raw_url
            try:
                parsed = urlparse(raw_url.replace("+asyncpg", ""))
                self.db_user = parsed.username or self.db_user
                self.db_password = parsed.password or self.db_password
                self.db_host = parsed.hostname or self.db_host
                self.db_port = str(parsed.port or self.db_port)
                self.db_name = parsed.path.lstrip("/") or self.db_name
            except Exception as parse_err:
                logger.error(f"Failed to extract DB metadata components from string: {str(parse_err)}")
        else:
            # Construct standard local URL out of the safe variables
            self.db_url = f"postgresql+asyncpg://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
            
        # 3. Create the Database Engine with a Connection Pool
        logger.info(f"Connecting Async Engine to Database URL: postgresql+asyncpg://{self.db_user}:***@{self.db_host}:{self.db_port}/{self.db_name}")
        
        self.engine = create_async_engine(
            self.db_url,
            echo=False,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True
        )
        
        # 4. Create a Session Factory
        self.session_factory = async_sessionmaker(
            bind=self.engine,
            autoflush=False,
            expire_on_commit=False,
            class_=AsyncSession
        )

    def get_psycopg_url(self) -> str:
        """
        Assembles a clean connection string compatible with native psycopg drivers.
        Drops the '+asyncpg' dialect prefix since psycopg requires standard URLs.
        """
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