from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import declarative_base
from app.core.load_property import settings
from app.observability.logger import logger, get_current_correlation_id

# The Base class that all your future Database Tables (Models) will inherit from
Base = declarative_base()

class DatabaseService:
    """
    Central service handling async PostgreSQL connection infrastructure.
    Manages connection pooling and provides isolated database sessions per request.
    """
    def __init__(self):
        # 1. Assemble the Async Connection URL for SQLAlchemy (requires +asyncpg driver)
        self.db_url = f"postgresql+asyncpg://{settings.db_user}:{settings.db_password}@{settings.db_host}:{settings.db_port}/{settings.db_name}"
        
        # 2. Create the Database Engine with a Connection Pool
        self.engine = create_async_engine(
            self.db_url,
            echo=False,               # Set to True if you want to see raw SQL text in logs
            pool_size=10,             # Keep 10 database connections open and ready to reuse
            max_overflow=20,          # Allow up to 20 extra connections if traffic spikes
            pool_pre_ping=True        # Automatically check if a connection is dead before using it
        )
        
        # 3. Create a Session Factory (The factory that mints transaction workers)
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
        return f"postgresql://{settings.db_user}:{settings.db_password}@{settings.db_host}:{settings.db_port}/{settings.db_name}"

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
            # Safely commit transactions if no errors occurred
            await session.commit()
        except Exception as e:
            # If anything goes wrong during a query, undo changes to keep database safe
            await session.rollback()
            log.error("Database transaction rolled back due to error", error=str(e))
            raise
        finally:
            # Always return the connection back to the pool
            await session.close()
            log.debug("Database session closed and returned to pool.")

    async def close(self):
        """Disconnect the engine pool when the application shuts down."""
        logger.info("Closing PostgreSQL database connection pool...")
        await self.engine.dispose()

# Create a singleton instance to manage the master pool
db_service = DatabaseService()