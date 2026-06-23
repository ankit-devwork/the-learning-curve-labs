import hashlib
from typing import Any

from pycorekit.exceptions.file import FileException
from supabase import Client

from app.core.auth import AuthUser
from app.core.cache import cache_get, cache_set, check_rate_limit
from app.core.exceptions import NotFoundException, RateLimitException
from app.core.yaml_config import get_yaml_config
from app.services.document_service import get_document_summary
from app.services.llm_client import audio_overview_cache_key, generate_audio_overview_script
from app.services.workspace_access import get_accessible_document, require_editable_document


async def generate_audio_overview(
    client: Client,
    document_id: str,
    user: AuthUser,
) -> dict[str, Any]:
    cfg = get_yaml_config().audio_overview
    allowed, retry_after = await check_rate_limit(
        key=f"audio_overview:{user.id}",
        limit=cfg.generate_rate_limit_per_min,
        window_seconds=60,
    )
    if not allowed:
        raise RateLimitException(
            f"Audio overview limit reached ({cfg.generate_rate_limit_per_min}/min)",
            retry_after=retry_after,
        )

    doc = require_editable_document(client, document_id, user)
    if doc["file_type"] != "document":
        raise FileException("Audio overviews are only available for document uploads")
    if doc["status"] != "ready":
        raise FileException("Document must be processed before generating an audio overview", status_code=409)

    summary_payload = await get_document_summary(client, document_id, user)
    summary = summary_payload.get("summary") or ""
    if not summary.strip():
        raise FileException("Document summary is required before generating an audio overview", status_code=409)

    content_hash = hashlib.sha256(summary.encode("utf-8")).hexdigest()[:16]
    cache_key = audio_overview_cache_key(user.id, document_id, content_hash)
    cached = await cache_get(cache_key)
    if cached and cached.get("script"):
        return {
            "document_id": document_id,
            "title": cached.get("title") or f"Audio overview: {doc['filename']}",
            "script": cached["script"],
            "estimated_minutes": cached.get("estimated_minutes"),
            "cached": True,
        }

    script = await generate_audio_overview_script(summary=summary, filename=doc["filename"])
    word_count = len(script.split())
    estimated_minutes = max(1, round(word_count / 150))
    payload = {
        "title": f"Audio overview: {doc['filename']}",
        "script": script,
        "estimated_minutes": estimated_minutes,
    }
    await cache_set(cache_key, payload, get_yaml_config().cache.artifact_ttl)
    return {
        "document_id": document_id,
        **payload,
        "cached": False,
    }


async def get_audio_overview(
    client: Client,
    document_id: str,
    user: AuthUser,
) -> dict[str, Any] | None:
    doc = get_accessible_document(client, document_id, user, min_role="viewer")
    if doc["status"] != "ready":
        return None

    summary_payload = await get_document_summary(client, document_id, user)
    summary = summary_payload.get("summary") or ""
    if not summary.strip():
        return None

    content_hash = hashlib.sha256(summary.encode("utf-8")).hexdigest()[:16]
    cache_key = audio_overview_cache_key(user.id, document_id, content_hash)
    cached = await cache_get(cache_key)
    if not cached or not cached.get("script"):
        return None

    return {
        "document_id": document_id,
        "title": cached.get("title") or f"Audio overview: {doc['filename']}",
        "script": cached["script"],
        "estimated_minutes": cached.get("estimated_minutes"),
        "cached": True,
    }
