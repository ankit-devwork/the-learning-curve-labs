import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row

# Core System Frameworks and Middleware Stack
from app.api.routes.observability import router as observability_router
from app.observability import ObservabilityMiddleware
from app.core.exception_handlers import app_exception_handler, generic_exception_handler
from app.core.exceptions import AppException
from app.core.database import db_service, Base
from app.core.graph_db import graph_service
from app.core.load_property import settings
from app.observability.logger import logger

# Routing Layers and Domain Services
from app.api.routes.document_ingestion import router as ingestion_router
from app.api.routes.query_engine import router as query_router
from app.services.cache_service import cache_service

# Unified Redis abstraction
from app.core.unified_redis import redis

# Models to ensure physical relational schema generation tracking
from app.models.document import DocumentModel
from app.models.document_chunk import DocumentChunkModel

# Checkpointer Components to compile the Durable Persistent Graph
from app.agents.agent_graph import workflow, AgentState
import app.agents.agent_graph as agent_module


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the operational lifecycle boundaries of the application.
    Orchestrates connection pooling initialization and persistence layer schema verification.
    """

    # ----------------------------------------------------------------------
    # 1. Initialize Postgres schema (documents, chunks, pgvector)
    # ----------------------------------------------------------------------
    async with db_service.engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables initialized successfully.")

    # ----------------------------------------------------------------------
    # 2. Initialize LangGraph Postgres Checkpointer
    # ----------------------------------------------------------------------
    try:
        db_url = db_service.get_psycopg_url()

        connection_kwargs = {
            "autocommit": True,
            "prepare_threshold": 0,
            "row_factory": dict_row,
        }

        agent_module.pool = AsyncConnectionPool(
            conninfo=db_url,
            max_size=getattr(settings, "db_pool_max_size", 10),
            open=False,
            kwargs=connection_kwargs
        )

        await agent_module.pool.open()

        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        postgres_checkpointer = AsyncPostgresSaver(agent_module.pool)

        await postgres_checkpointer.setup()
        agent_module.graph_rag_executor = workflow.compile(checkpointer=postgres_checkpointer)

        logger.info("LangGraph Postgres checkpointer verified, migrated, and compiled successfully.")

    except Exception as pg_sync_err:
        logger.error(f"Failed to wire durable LangGraph checkpointer storage context: {pg_sync_err}")

    # ----------------------------------------------------------------------
    # Startup complete — yield control to FastAPI
    # ----------------------------------------------------------------------
    yield

    # ----------------------------------------------------------------------
    # SHUTDOWN SEQUENCE — gracefully close all resources
    # ----------------------------------------------------------------------
    logger.info("Draining active persistence resource connection pool handlers...")

    # Close LangGraph checkpointer pool
    if hasattr(agent_module, "pool") and agent_module.pool:
        await agent_module.pool.close()
        logger.info("LangGraph checkpointer connection pool closed.")

    # Close Unified Redis
    await redis.close()

    # Close Postgres SQLAlchemy engine
    await db_service.close()

    # Close Neo4j driver
    await graph_service.close()

    logger.info("Application infrastructure connections torn down cleanly.")


def create_app():
    """Compiles global settings, route schemas, error proxies, and tracking middleware blocks."""
    app = FastAPI(title="Digital Worker Studio", lifespan=lifespan)

    # Global Monitoring and Performance Evaluation Middleware
    app.add_middleware(ObservabilityMiddleware)

    # Global Exception Catching Triggers
    app.add_exception_handler(AppException, app_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)

    # Router Component Encasement Mapping
    app.include_router(observability_router)
    app.include_router(query_router)
    app.include_router(ingestion_router)

    return app


app = create_app()
