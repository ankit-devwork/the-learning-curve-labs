import os
import time
from fastapi import APIRouter, Depends
from litellm import acompletion
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from neo4j.exceptions import ServiceUnavailable, Neo4jError
from neo4j import AsyncSession as Neo4jAsyncSession 

from app.core.load_property import settings
from app.core.database import db_service
from app.core.redis import redis_service
from app.core.exceptions import (
    DatabaseConnectionException, 
    RedisConnectionException, 
    Neo4jConnectionException, 
    GraphQueryException
)
from app.core.graph_db import graph_service
from app.observability.decorators import with_observability
from app.observability.logger import logger
from app.observability.worker import get_worker_logger

# Define a uniform API prefix path grouping
router = APIRouter(prefix="/api/monitor", tags=["Monitoring & Diagnostics"])


@with_observability(
    name="observability_test_handler",
    include_args=True,
    include_result=True,
    model=settings.base_model,
)
async def run_test_logic(prompt: str):
    """
    Asynchronously executes LLM generation using LiteLLM.
    Natively routes with the global, sanitized model string provider prefix.
    """
    return await acompletion(
        model=settings.base_model,  # 🚀 Clean and streamlined: e.g., 'groq/llama-3.3-70b-versatile'
        messages=[{"role": "user", "content": prompt}]
    )


@router.get("/observability-check")
async def observability_test():
    logger.info("Starting observability test endpoint")

    worker_log = get_worker_logger()
    worker_log.info("Worker logger is functioning inside API test context")

    model_response = await run_test_logic(
        prompt="Say hello. This is an observability test."
    )

    logger.info("Observability test completed")
    return {
        "status": "ok",
        "model_response": model_response,
    }


@router.get("/db-check")
async def check_postgres_health(db: AsyncSession = Depends(db_service.get_session)):
    """Validates the availability and responsiveness of the PostgreSQL connection pool."""
    try:
        result = await db.execute(text("SELECT 1"))
        result.scalar()
        return {"status": "operational", "database": "connected"}
    except Exception as e:
        raise DatabaseConnectionException(
            message="PostgreSQL connection check failed.",
            details={"raw_error": str(e)}
        )


@router.get("/redis-check")
async def check_redis_health():
    """Validates the responsiveness of the Redis cache engine."""
    try:
        is_alive = await redis_service.client.ping()
        if is_alive:
            return {"status": "operational", "cache": "connected"}
    except Exception as e:
        raise RedisConnectionException(
            message="Redis connection check failed.",
            details={"raw_error": str(e)}
        )


@router.get("/neo4j-check")
async def check_neo4j_health():
    """
    Pings the local Neo4j Graph Database to ensure connection pooling 
    and authentication credentials are fully functional.
    """
    driver = graph_service.get_driver()
    start_time = time.perf_counter()
    
    try:
        # Clear out 'session_cls' completely — AsyncDriver handles this natively
        async with driver.session() as session:
            # 'RETURN 1' acts as our graph ping
            result = await session.run("RETURN 1 AS ping")
            record = await result.single()
            
            if record and record["ping"] == 1:
                latency_ms = (time.perf_counter() - start_time) * 1000
                return {
                    "status": "operational",
                    "database": "connected",
                    "latency_ms": round(latency_ms, 2),
                    "message": "Graph engine connection pool is green and operational."
                }
            else:
                raise GraphQueryException(
                    message="Neo4j responded, but returned an unexpected or empty payload structure."
                )
                
    except ServiceUnavailable as conn_error:
        logger.error(f"Neo4j Health Check Service Unreachable: {str(conn_error)}")
        raise Neo4jConnectionException(
            message="Graph database service layer is currently unreachable.",
            details={"error_class": "ServiceUnavailable", "raw_message": str(conn_error)}
        )
        
    except Neo4jError as query_error:
        logger.error(f"Neo4j Query execution exception: {str(query_error)}")
        raise GraphQueryException(
            message="Graph server accepted connection but rejected the ping transaction.",
            details={"neo4j_code": query_error.code, "raw_message": query_error.message}
        )
        
    except Exception as unexpected_error:
        logger.error(f"Unexpected health check failure: {str(unexpected_error)}")
        raise Neo4jConnectionException(
            message="An unexpected issue occurred while pinging the graph database tier.",
            details={"raw_message": str(unexpected_error)}
        )