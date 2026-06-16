import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pycorekit.core_logging.logger import init_logger
from pycorekit.exceptions.base import AppException
from pycorekit.exceptions.handlers import app_exception_handler, generic_exception_handler
from pycorekit.tracing.middleware import RequestTracingMiddleware

from app.api.routes.health import router as health_router
from app.api.routes.me import router as me_router
from app.api.routes.upload import router as upload_router
from app.core.config import settings
from app.core.neo4j_client import neo4j_client
from app.core.redis_client import redis_client

LOG_DIR = os.getenv("LOG_DIR", "logs")

init_logger(
    log_dir=LOG_DIR,
    rotation="00:00",
    retention="7 days",
    json_file=True,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await redis_client.close()
    await neo4j_client.close()


app = FastAPI(
    title=settings.app_name,
    description="InsightLab — Excel insights, document chat, and AI quizzes",
    version="0.2.0",
    lifespan=lifespan,
)

_cors_origins = os.getenv(
    "CORS_ALLOW_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in _cors_origins.split(",") if origin.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestTracingMiddleware)

app.add_exception_handler(AppException, app_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)

app.include_router(health_router, tags=["health"])
app.include_router(me_router, tags=["auth"])
app.include_router(upload_router, tags=["documents"])


@app.get("/")
async def root():
    return {
        "name": "InsightLab API",
        "version": "0.2.0",
        "docs": "/docs",
        "health": "/health",
        "me": "/me",
        "upload": "/upload",
        "documents": "/documents",
    }
