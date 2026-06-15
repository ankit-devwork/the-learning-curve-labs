from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health():
    return {"status": "ok", "service": "insightlab-api"}


@router.get("/ready")
async def ready():
    # Phase 1: ping Redis, Neo4j, Supabase, LLM key presence
    return {
        "status": "ready",
        "checks": {
            "api": True,
            "redis": "not_configured",
            "neo4j": "not_configured",
            "supabase": "not_configured",
        },
    }
