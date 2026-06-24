import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pycorekit.core_logging.logger import get_logger, init_logger
from pycorekit.exceptions.base import AppException
from pycorekit.exceptions.handlers import app_exception_handler, generic_exception_handler
from pycorekit.tracing.middleware import RequestTracingMiddleware

from app.api.routes.artifacts import router as artifacts_router
from app.api.routes.graph import router as graph_router
from app.api.routes.workspaces import router as workspaces_router
from app.api.routes.documents import router as documents_router
from app.api.routes.excel import router as excel_router
from app.api.routes.invites import router as invites_router
from app.api.routes.health import router as health_router
from app.api.routes.me import router as me_router
from app.api.routes.upload import router as upload_router
from app.api.routes.study_sessions import router as study_sessions_router
from app.api.routes.public_quiz import router as public_quiz_router
from app.api.routes.quiz import router as quiz_router
from app.core.cache import close_cache
from app.core.config import ENV_PATH, config_diagnostics, settings
from app.core.security import is_production, validate_cors_origins_at_startup
from app.core.yaml_config import get_yaml_config
from app.core.neo4j_client import neo4j_client
from app.core.redis_client import redis_client

LOG_DIR = os.getenv("LOG_DIR", get_yaml_config().logging.dir)

init_logger(
    log_dir=LOG_DIR,
    rotation="00:00",
    retention="7 days",
    json_file=True,
)

if not settings.supabase_url.strip():
    diag = config_diagnostics()
    get_logger("startup").warning(
        "SUPABASE_URL is not loaded — authenticated routes will return 401 for ES256 tokens. "
        f"env_file={diag['env_file']} exists={diag['env_file_exists']} "
        f"root_env_exists={diag['root_env_file_exists']}. "
        "If backend/.env has the value, check for an empty SUPABASE_URL in Conda/shell env "
        "or save .env as UTF-8 (not UTF-16).",
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await close_cache()
    await redis_client.close()
    await neo4j_client.close()


validate_cors_origins_at_startup()

_show_api_docs = not is_production()

app = FastAPI(
    title=get_yaml_config().app.name,
    description="InsightLab — Excel insights, document chat, and AI quizzes",
    version="0.2.0",
    lifespan=lifespan,
    docs_url="/docs" if _show_api_docs else None,
    redoc_url="/redoc" if _show_api_docs else None,
    openapi_url="/openapi.json" if _show_api_docs else None,
)

_cors_origins = os.getenv(
    "CORS_ALLOW_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in _cors_origins.split(",") if origin.strip()],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=[
        "Authorization",
        "Content-Type",
        "X-Request-ID",
        "X-Tracking-ID",
        "X-Correlation-Id",
        "x-correlation-id",
    ],
    expose_headers=["X-Tracking-ID", "X-Correlation-Id", "x-correlation-id"],
)
app.add_middleware(RequestTracingMiddleware)

app.add_exception_handler(AppException, app_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)

app.include_router(health_router, tags=["health"])
app.include_router(me_router, tags=["auth"])
app.include_router(upload_router, tags=["documents"])
app.include_router(documents_router, tags=["documents"])
app.include_router(workspaces_router)
app.include_router(invites_router)
app.include_router(artifacts_router)
app.include_router(graph_router, tags=["graph"])
app.include_router(excel_router, tags=["excel"])
app.include_router(quiz_router, tags=["quiz"])
app.include_router(public_quiz_router)
app.include_router(study_sessions_router)


@app.get("/")
async def root():
    payload = {
        "name": "InsightLab API",
        "version": "0.2.0",
        "health": "/health",
    }
    if _show_api_docs:
        payload["docs"] = "/docs"
    return payload
