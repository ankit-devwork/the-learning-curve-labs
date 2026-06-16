from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.health import router as health_router
from app.core.config import settings
from app.core.neo4j_client import neo4j_client
from app.core.redis_client import redis_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await redis_client.close()
    await neo4j_client.close()


app = FastAPI(
    title=settings.app_name,
    description="InsightLab — Excel insights, document chat, and AI quizzes",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router, tags=["health"])


@app.get("/")
async def root():
    return {
        "name": "InsightLab API",
        "docs": "/docs",
        "health": "/health",
    }
