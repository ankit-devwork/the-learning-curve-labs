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
from app.core.redis import redis_service 
from app.core.database import db_service, Base
from app.core.graph_db import graph_service
from app.core.load_property import settings
from app.observability.logger import logger

# Routing Layers and Domain Services
from app.api.routes.document_ingestion import router as ingestion_router
from app.api.routes.query_engine import router as query_router
from app.services.cache_service import cache_service

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
    # 1. Physicalize Postgres Relational & pgvector Data Tables
    async with db_service.engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables initialized successfully.")

    # 2. Configure and Instantiate the Durable LangGraph Postgres Checkpointer
    try:
        # Pull the connection string from the new helper function in db_service
        db_url = db_service.get_psycopg_url()

        connection_kwargs = {
            "autocommit": True,
            "prepare_threshold": 0,
            "row_factory": dict_row,  # Enforces mapping underlying storage tuples to native dicts
        }

        # 🚀 FIX: Pass open=False to prevent the deprecated initialization pattern inside the constructor
        agent_module.pool = AsyncConnectionPool(
            conninfo=db_url,
            max_size=settings.db_pool_max_size if hasattr(settings, "db_pool_max_size") else 10,
            open=False,
            kwargs=connection_kwargs
        )
        
        # 🚀 FIX: Explicitly and asynchronously open the connection pool before passing it to LangGraph
        await agent_module.pool.open()

        # Map the checkpointer instance to the freshly opened pool context
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        postgres_checkpointer = AsyncPostgresSaver(agent_module.pool)

        # Setup state tables if they don't already exist and compile the graph
        await postgres_checkpointer.setup()
        agent_module.graph_rag_executor = workflow.compile(checkpointer=postgres_checkpointer)
        logger.info("LangGraph Postgres checkpointer verified, migrated, and compiled successfully.")

    except Exception as pg_sync_err:
        logger.error(f"Failed to wire durable LangGraph checkpointer storage context: {pg_sync_err}")

    # 3. Initialize High-Performance Redis Response Cache
    try:
        cache_service.initialize()
        logger.info("Async Redis Response Cache Engine initialized successfully.")
    except Exception as cache_err:
        logger.error(f"Failed to wire Redis connection loop during application lifespan boot: {cache_err}")

    # Startup sequence finalized; yield execution control to incoming HTTP application worker threads
    yield

    # -----------------------------------------------------------------------------------------
    # SHUTDOWN PROCESS SEQUENCE (Gracefully draining resource connection pools)
    # -----------------------------------------------------------------------------------------
    logger.info("Draining active persistence resource connection pool handlers...")
    
    # Close checkpointer pool connections if initialized
    if hasattr(agent_module, "pool") and agent_module.pool:
        await agent_module.pool.close()
        logger.info("LangGraph checkpointer connection pool closed.")
        
    # Close relational, graph, and core cache engine connections
    await redis_service.close()
    await db_service.close()
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