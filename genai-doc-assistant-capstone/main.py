"""
Main entrypoint for the GenAI Document Assistant Capstone Project.

Responsibilities:
- Initialize logging (via pycorekit)
- Load configuration
- Register API routes
- Add tracing + correlation ID middleware
- Start FastAPI application
"""

from fastapi import FastAPI

# pycorekit integrations
from pycorekit.core_logging.logger import init_logger
from pycorekit.exceptions.base import AppException
from pycorekit.exceptions.handlers import app_exception_handler, generic_exception_handler
# Local imports
from pycorekit.tracing.middleware import RequestTracingMiddleware
from app.api.routes_health import router as health_router
from app.api.routes_upload import router as upload_router
from app.api.routes_query import router as query_router
from app.api.routes_choose_document import router as choose_doc_router
from app.api.routes_list_documents import router as list_doc_router
from app.api.routes_observability import router as observability_router
from app.core.settings import settings
# Initialize logging
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


init_logger(
    log_dir=settings.paths.log_dir,
    rotation="00:00",
    retention="7 days",
    json_file=True
)

# FastAPI app
app = FastAPI(
    title="GenAI Document Assistant - Capstone",
    version="1.0.0",
    description="A document ingestion + RAG-based assistant built for the capstone project."
)

# Add middleware

app.add_middleware(RequestTracingMiddleware)

app.add_exception_handler(AppException, app_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)

# Register routes
app.include_router(health_router)
app.include_router(upload_router)
app.include_router(query_router)
app.include_router(choose_doc_router)
app.include_router(list_doc_router)
app.include_router(observability_router)