import hashlib
from typing import Any
from uuid import uuid4

from pycorekit.exceptions.file import FileException
from supabase import Client

from app.core.auth import AuthUser
from app.core.cache import cache_get, cache_set, check_rate_limit
from app.core.exceptions import NotFoundException, RateLimitException
from app.core.migration_guard import is_missing_phase14_schema, run_or_none_phase14
from app.core.yaml_config import get_yaml_config
from app.services.document_service import get_document_summary
from app.services.llm_client import audio_overview_cache_key, generate_audio_overview_script
from app.services.tts_service import synthesize_mp3, tts_voice_from_env
from app.services.workspace_access import get_accessible_document, require_editable_document


def _serialize_overview(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "overview_id": row["id"],
        "document_id": row["document_id"],
        "title": row["title"],
        "script": row["script"],
        "estimated_minutes": row.get("estimated_minutes"),
        "has_audio": bool(row.get("storage_path")),
        "created_at": row.get("created_at"),
    }


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

    existing = await get_document_audio_overview(client, document_id, user)
    if existing:
        return {**existing, "cached": True}

    content_hash = hashlib.sha256(summary.encode("utf-8")).hexdigest()[:16]
    cache_key = audio_overview_cache_key(user.id, document_id, content_hash)
    cached = await cache_get(cache_key)
    script = cached.get("script") if cached else None
    if not script:
        script = await generate_audio_overview_script(summary=summary, filename=doc["filename"])

    word_count = len(script.split())
    estimated_minutes = max(1, round(word_count / 150))
    title = f"Audio overview: {doc['filename']}"
    overview_id = str(uuid4())
    storage_path: str | None = None

    if cfg.tts_enabled:
        try:
            mp3_bytes = await synthesize_mp3(script, voice=tts_voice_from_env())
            upload_cfg = get_yaml_config().upload
            storage_path = f"{user.id}/audio/{document_id}/{overview_id}.mp3"
            client.storage.from_(upload_cfg.storage_bucket).upload(
                storage_path,
                mp3_bytes,
                file_options={"content-type": "audio/mpeg", "upsert": "true"},
            )
        except Exception:
            storage_path = None

    row = {
        "id": overview_id,
        "document_id": document_id,
        "workspace_id": doc["workspace_id"],
        "owner_id": user.id,
        "title": title,
        "script": script,
        "storage_path": storage_path,
        "estimated_minutes": estimated_minutes,
    }

    def _insert() -> dict[str, Any]:
        inserted = client.table("document_audio_overviews").insert(row).execute().data or []
        return inserted[0] if inserted else row

    saved = run_or_none_phase14(_insert)
    if saved is None:
        await cache_set(
            cache_key,
            {"script": script, "title": title, "estimated_minutes": estimated_minutes},
            get_yaml_config().cache.artifact_ttl,
        )
        return {
            "document_id": document_id,
            "title": title,
            "script": script,
            "estimated_minutes": estimated_minutes,
            "has_audio": False,
            "cached": False,
        }

    await cache_set(cache_key, {"overview_id": overview_id}, get_yaml_config().cache.artifact_ttl)
    return {**_serialize_overview(saved), "cached": False}


async def get_document_audio_overview(
    client: Client,
    document_id: str,
    user: AuthUser,
) -> dict[str, Any] | None:
    doc = get_accessible_document(client, document_id, user, min_role="viewer")
    if doc["status"] != "ready":
        return None

    def _load() -> dict[str, Any] | None:
        rows = (
            client.table("document_audio_overviews")
            .select("*")
            .eq("document_id", document_id)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
            .data
            or []
        )
        return rows[0] if rows else None

    row = run_or_none_phase14(_load)
    if row:
        return _serialize_overview(row)

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
        "has_audio": False,
        "cached": True,
    }


async def get_audio_overview_mp3_bytes(
    client: Client,
    document_id: str,
    user: AuthUser,
) -> tuple[bytes, str]:
    overview = await get_document_audio_overview(client, document_id, user)
    if not overview:
        raise NotFoundException("Audio overview not found")

    overview_id = overview.get("overview_id")
    if not overview_id:
        raise FileException("Generate audio overview first", status_code=404)

    def _load_path() -> str | None:
        rows = (
            client.table("document_audio_overviews")
            .select("storage_path, title")
            .eq("id", overview_id)
            .limit(1)
            .execute()
            .data
            or []
        )
        if not rows:
            return None
        return rows[0].get("storage_path")

    storage_path = run_or_none_phase14(_load_path)
    if not storage_path:
        raise FileException("Audio file not available — regenerate the overview", status_code=404)

    upload_cfg = get_yaml_config().upload
    data = client.storage.from_(upload_cfg.storage_bucket).download(storage_path)
    filename = overview.get("title") or "audio-overview"
    safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in filename)[:80]
    return data, f"{safe_name}.mp3"


# Backward-compatible alias
async def get_audio_overview(
    client: Client,
    document_id: str,
    user: AuthUser,
) -> dict[str, Any] | None:
    return await get_document_audio_overview(client, document_id, user)
