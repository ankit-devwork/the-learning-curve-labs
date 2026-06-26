from typing import Any
from uuid import uuid4

from pycorekit.core_logging.logger import get_logger
from pycorekit.exceptions.file import FileException
from supabase import Client

from app.core.auth import AuthUser
from app.core.cache import cache_get, cache_set, check_rate_limit
from app.core.migration_guard import PHASE3_020_MIGRATION_NOTICE, run_or_none_phase14
from app.core.exceptions import NotFoundException, RateLimitException
from app.core.safe_errors import sanitize_stored_error
from app.core.yaml_config import get_yaml_config
from app.services.artifact_parsing import (
    flashcard_draft_to_rows,
    infographic_to_content,
    parse_flashcard_draft,
    parse_infographic_draft,
    parse_slide_deck_draft,
    parse_study_guide_draft,
    slide_deck_to_content,
    study_guide_to_content,
)
from app.services.flashcard_srs_service import get_due_flashcard_ids, _update_srs_safe
from app.services.llm_client import (
    flashcard_cache_key,
    generate_flashcard_draft,
    generate_infographic_draft,
    generate_slide_deck_draft,
    generate_study_guide_draft,
    infographic_cache_key,
    slide_deck_cache_key,
    study_guide_cache_key,
)
from app.services.workspace_access import get_accessible_document, require_editable_document
from app.services.quiz_service import _sample_chunks

log = get_logger("artifacts")


async def generate_document_flashcards(
    client: Client,
    document_id: str,
    user: AuthUser,
    *,
    num_cards: int = 10,
) -> dict[str, Any]:
    cfg = get_yaml_config().artifacts
    allowed, retry_after = await check_rate_limit(
        key=f"flashcards:{user.id}",
        limit=cfg.generate_rate_limit_per_min,
        window_seconds=60,
    )
    if not allowed:
        raise RateLimitException(
            f"Flashcard generation limit reached ({cfg.generate_rate_limit_per_min}/min)",
            retry_after=retry_after,
        )

    num_cards = min(max(num_cards, 3), cfg.max_flashcards)
    cache_key = flashcard_cache_key(user.id, document_id, num_cards)
    cached = await cache_get(cache_key)
    if cached and cached.get("set_id"):
        return await get_flashcard_set(client, cached["set_id"], user)

    doc = require_editable_document(client, document_id, user)
    if doc["file_type"] != "document":
        raise FileException("Flashcards are only available for document uploads")
    if doc["status"] != "ready":
        raise FileException("Document must be processed before generating flashcards", status_code=409)

    chunks = (
        client.table("document_chunks")
        .select("chunk_index, content")
        .eq("document_id", document_id)
        .order("chunk_index")
        .execute()
        .data
        or []
    )
    if not chunks:
        raise FileException("No document chunks found; reprocess the document", status_code=409)

    sampled = _sample_chunks(chunks, cfg.max_context_chunks)
    chunk_indexes = [row["chunk_index"] for row in sampled]
    context_chunks = [row["content"] for row in sampled]

    raw = await generate_flashcard_draft(
        context_chunks=context_chunks,
        filename=doc["filename"],
        num_cards=num_cards,
    )
    try:
        draft = parse_flashcard_draft(raw, max_cards=num_cards)
    except (ValueError, Exception) as exc:
        raise FileException(f"Invalid flashcard payload from LLM: {exc}", status_code=502) from exc

    set_id = str(uuid4())
    card_rows = flashcard_draft_to_rows(draft, chunk_indexes=chunk_indexes)
    for row in card_rows:
        row["id"] = str(uuid4())
        row["set_id"] = set_id

    client.table("flashcard_sets").insert(
        {
            "id": set_id,
            "document_id": document_id,
            "workspace_id": doc["workspace_id"],
            "owner_id": user.id,
            "title": draft.title.strip() or f"Flashcards: {doc['filename']}",
        }
    ).execute()
    client.table("flashcards").insert(card_rows).execute()
    await cache_set(cache_key, {"set_id": set_id}, get_yaml_config().cache.artifact_ttl)
    log.info("Flashcards generated", set_id=set_id, document_id=document_id, user_id=user.id)
    return await get_flashcard_set(client, set_id, user)


async def get_document_flashcards(client: Client, document_id: str, user: AuthUser) -> dict[str, Any] | None:
    get_accessible_document(client, document_id, user, min_role="viewer")
    result = (
        client.table("flashcard_sets")
        .select("id")
        .eq("document_id", document_id)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    if not result.data:
        return None
    return await get_flashcard_set(client, result.data[0]["id"], user)


async def get_flashcard_set(client: Client, set_id: str, user: AuthUser) -> dict[str, Any]:
    set_result = (
        client.table("flashcard_sets")
        .select("*")
        .eq("id", set_id)
        .limit(1)
        .execute()
    )
    if not set_result.data:
        raise NotFoundException("Flashcard set not found")
    flashcard_set = set_result.data[0]
    get_accessible_document(client, flashcard_set["document_id"], user, min_role="viewer")
    cards = (
        client.table("flashcards")
        .select("id, front, back, sort_order, source_chunk_index")
        .eq("set_id", set_id)
        .order("sort_order")
        .execute()
        .data
        or []
    )
    card_ids = [row["id"] for row in cards]
    srs = run_or_none_phase14(lambda: get_due_flashcard_ids(client, user, flashcard_ids=card_ids))
    due_count = srs["due_count"] if srs else len(cards)
    due_ids = srs["due_ids"] if srs else card_ids
    return {
        "set_id": flashcard_set["id"],
        "document_id": flashcard_set["document_id"],
        "title": flashcard_set["title"],
        "cards": cards,
        "card_count": len(cards),
        "due_count": due_count,
        "due_ids": due_ids,
    }


async def review_flashcard(
    client: Client,
    set_id: str,
    user: AuthUser,
    *,
    flashcard_id: str,
    knew: bool,
) -> dict[str, Any]:
    cfg = get_yaml_config().artifacts
    allowed, retry_after = await check_rate_limit(
        key=f"flashcard_review:{user.id}",
        limit=cfg.review_rate_limit_per_min,
        window_seconds=60,
    )
    if not allowed:
        raise RateLimitException(
            f"Flashcard review limit reached ({cfg.review_rate_limit_per_min}/min)",
            retry_after=retry_after,
        )

    await get_flashcard_set(client, set_id, user)
    card = (
        client.table("flashcards")
        .select("id")
        .eq("id", flashcard_id)
        .eq("set_id", set_id)
        .limit(1)
        .execute()
    )
    if not card.data:
        raise NotFoundException("Flashcard not found")

    client.table("flashcard_reviews").insert(
        {
            "flashcard_id": flashcard_id,
            "user_id": user.id,
            "knew": knew,
        }
    ).execute()
    srs = _update_srs_safe(client, user, flashcard_id=flashcard_id, knew=knew)
    return {
        "flashcard_id": flashcard_id,
        "knew": knew,
        "recorded": True,
        "srs": srs,
    }


async def generate_document_study_guide(
    client: Client,
    document_id: str,
    user: AuthUser,
) -> dict[str, Any]:
    cfg = get_yaml_config().artifacts
    allowed, retry_after = await check_rate_limit(
        key=f"study_guide:{user.id}",
        limit=cfg.generate_rate_limit_per_min,
        window_seconds=60,
    )
    if not allowed:
        raise RateLimitException(
            f"Study guide generation limit reached ({cfg.generate_rate_limit_per_min}/min)",
            retry_after=retry_after,
        )

    cache_key = study_guide_cache_key(user.id, document_id)
    cached = await cache_get(cache_key)
    if cached and cached.get("guide_id"):
        return await get_study_guide_by_id(client, cached["guide_id"], user)

    doc = require_editable_document(client, document_id, user)
    if doc["file_type"] != "document":
        raise FileException("Study guides are only available for document uploads")
    if doc["status"] != "ready" or not doc.get("summary"):
        raise FileException("Document must be processed before generating a study guide", status_code=409)

    chunks = (
        client.table("document_chunks")
        .select("chunk_index, content")
        .eq("document_id", document_id)
        .order("chunk_index")
        .execute()
        .data
        or []
    )
    sampled = _sample_chunks(chunks, cfg.max_context_chunks)
    context_chunks = [row["content"] for row in sampled]

    raw = await generate_study_guide_draft(
        context_chunks=context_chunks,
        summary=doc["summary"],
        filename=doc["filename"],
    )
    try:
        draft = parse_study_guide_draft(raw)
    except (ValueError, Exception) as exc:
        raise FileException(f"Invalid study guide payload from LLM: {exc}", status_code=502) from exc

    guide_id = str(uuid4())
    content = study_guide_to_content(draft)
    client.table("study_guides").insert(
        {
            "id": guide_id,
            "document_id": document_id,
            "workspace_id": doc["workspace_id"],
            "owner_id": user.id,
            "title": content["title"] or f"Study guide: {doc['filename']}",
            "content": content,
        }
    ).execute()
    await cache_set(cache_key, {"guide_id": guide_id}, get_yaml_config().cache.artifact_ttl)
    log.info("Study guide generated", guide_id=guide_id, document_id=document_id, user_id=user.id)
    return await get_study_guide_by_id(client, guide_id, user)


async def get_document_study_guide(client: Client, document_id: str, user: AuthUser) -> dict[str, Any] | None:
    get_accessible_document(client, document_id, user, min_role="viewer")
    result = (
        client.table("study_guides")
        .select("id")
        .eq("document_id", document_id)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    if not result.data:
        return None
    return await get_study_guide_by_id(client, result.data[0]["id"], user)


async def get_study_guide_by_id(client: Client, guide_id: str, user: AuthUser) -> dict[str, Any]:
    result = (
        client.table("study_guides")
        .select("*")
        .eq("id", guide_id)
        .limit(1)
        .execute()
    )
    if not result.data:
        raise NotFoundException("Study guide not found")
    row = result.data[0]
    get_accessible_document(client, row["document_id"], user, min_role="viewer")
    return {
        "guide_id": row["id"],
        "document_id": row["document_id"],
        "title": row["title"],
        "content": row["content"],
        "created_at": row["created_at"],
    }


async def generate_document_infographic(
    client: Client,
    document_id: str,
    user: AuthUser,
) -> dict[str, Any]:
    cfg = get_yaml_config().artifacts
    allowed, retry_after = await check_rate_limit(
        key=f"infographic:{user.id}",
        limit=cfg.generate_rate_limit_per_min,
        window_seconds=60,
    )
    if not allowed:
        raise RateLimitException(
            f"Infographic generation limit reached ({cfg.generate_rate_limit_per_min}/min)",
            retry_after=retry_after,
        )

    cache_key = infographic_cache_key(user.id, document_id)
    cached = await cache_get(cache_key)
    if cached and cached.get("infographic_id"):
        return await get_infographic_by_id(client, cached["infographic_id"], user)

    doc = require_editable_document(client, document_id, user)
    if doc["file_type"] != "document":
        raise FileException("Infographics are only available for document uploads")
    if doc["status"] != "ready" or not doc.get("summary"):
        raise FileException("Document must be processed before generating an infographic", status_code=409)

    chunks = (
        client.table("document_chunks")
        .select("chunk_index, content")
        .eq("document_id", document_id)
        .order("chunk_index")
        .execute()
        .data
        or []
    )
    sampled = _sample_chunks(chunks, cfg.max_context_chunks)
    context_chunks = [row["content"] for row in sampled]

    raw = await generate_infographic_draft(
        context_chunks=context_chunks,
        summary=doc["summary"],
        filename=doc["filename"],
    )
    try:
        draft = parse_infographic_draft(raw)
        content = infographic_to_content(draft)
    except (ValueError, Exception) as exc:
        raise FileException(f"Invalid infographic payload from LLM: {exc}", status_code=502) from exc

    infographic_id = str(uuid4())
    client.table("document_infographics").insert(
        {
            "id": infographic_id,
            "document_id": document_id,
            "workspace_id": doc["workspace_id"],
            "owner_id": user.id,
            "title": content["title"] or f"Infographic: {doc['filename']}",
            "content": content,
        }
    ).execute()
    await cache_set(cache_key, {"infographic_id": infographic_id}, get_yaml_config().cache.artifact_ttl)
    log.info("Infographic generated", infographic_id=infographic_id, document_id=document_id, user_id=user.id)
    return await get_infographic_by_id(client, infographic_id, user)


async def get_document_infographic(client: Client, document_id: str, user: AuthUser) -> dict[str, Any] | None:
    get_accessible_document(client, document_id, user, min_role="viewer")
    result = (
        client.table("document_infographics")
        .select("id")
        .eq("document_id", document_id)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    if not result.data:
        return None
    return await get_infographic_by_id(client, result.data[0]["id"], user)


async def get_infographic_by_id(client: Client, infographic_id: str, user: AuthUser) -> dict[str, Any]:
    result = (
        client.table("document_infographics")
        .select("*")
        .eq("id", infographic_id)
        .limit(1)
        .execute()
    )
    if not result.data:
        raise NotFoundException("Infographic not found")
    row = result.data[0]
    get_accessible_document(client, row["document_id"], user, min_role="viewer")
    return {
        "infographic_id": row["id"],
        "document_id": row["document_id"],
        "title": row["title"],
        "content": row["content"],
        "created_at": row["created_at"],
    }


async def generate_document_slide_deck(
    client: Client,
    document_id: str,
    user: AuthUser,
) -> dict[str, Any]:
    cfg = get_yaml_config().artifacts
    allowed, retry_after = await check_rate_limit(
        key=f"slide_deck:{user.id}",
        limit=cfg.generate_rate_limit_per_min,
        window_seconds=60,
    )
    if not allowed:
        raise RateLimitException(
            f"Slide deck generation limit reached ({cfg.generate_rate_limit_per_min}/min)",
            retry_after=retry_after,
        )

    cache_key = slide_deck_cache_key(user.id, document_id)
    cached = await cache_get(cache_key)
    if cached and cached.get("slide_deck_id"):
        return await get_slide_deck_by_id(client, cached["slide_deck_id"], user)

    doc = require_editable_document(client, document_id, user)
    if doc["file_type"] != "document":
        raise FileException("Slide decks are only available for document uploads")
    if doc["status"] != "ready" or not doc.get("summary"):
        raise FileException("Document must be processed before generating a slide deck", status_code=409)

    chunks = (
        client.table("document_chunks")
        .select("chunk_index, content")
        .eq("document_id", document_id)
        .order("chunk_index")
        .execute()
        .data
        or []
    )
    sampled = _sample_chunks(chunks, cfg.max_context_chunks)
    context_chunks = [row["content"] for row in sampled]

    raw = await generate_slide_deck_draft(
        context_chunks=context_chunks,
        summary=doc["summary"],
        filename=doc["filename"],
        max_slides=cfg.max_slides,
    )
    try:
        draft = parse_slide_deck_draft(raw, max_slides=cfg.max_slides)
        content = slide_deck_to_content(draft)
    except (ValueError, Exception) as exc:
        raise FileException(f"Invalid slide deck payload from LLM: {exc}", status_code=502) from exc

    deck_id = str(uuid4())

    def _insert() -> dict[str, Any]:
        client.table("document_slide_decks").insert(
            {
                "id": deck_id,
                "document_id": document_id,
                "workspace_id": doc["workspace_id"],
                "owner_id": user.id,
                "title": content["title"] or f"Slides: {doc['filename']}",
                "content": content,
            }
        ).execute()
        return {"id": deck_id, "content": content}

    inserted = run_or_none_phase14(_insert)
    if inserted is None:
        raise FileException(PHASE3_020_MIGRATION_NOTICE, status_code=503)

    await cache_set(cache_key, {"slide_deck_id": deck_id}, get_yaml_config().cache.artifact_ttl)
    log.info("Slide deck generated", slide_deck_id=deck_id, document_id=document_id, user_id=user.id)
    return await get_slide_deck_by_id(client, deck_id, user)


async def get_document_slide_deck(client: Client, document_id: str, user: AuthUser) -> dict[str, Any] | None:
    get_accessible_document(client, document_id, user, min_role="viewer")

    def _load() -> dict[str, Any] | None:
        rows = (
            client.table("document_slide_decks")
            .select("id")
            .eq("document_id", document_id)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
            .data
            or []
        )
        return rows[0] if rows else None

    row = run_or_none_phase14(_load)
    if not row:
        return None
    return await get_slide_deck_by_id(client, row["id"], user)


async def get_slide_deck_by_id(client: Client, slide_deck_id: str, user: AuthUser) -> dict[str, Any]:
    result = (
        client.table("document_slide_decks")
        .select("*")
        .eq("id", slide_deck_id)
        .limit(1)
        .execute()
    )
    if not result.data:
        raise NotFoundException("Slide deck not found")
    row = result.data[0]
    get_accessible_document(client, row["document_id"], user, min_role="viewer")
    return {
        "slide_deck_id": row["id"],
        "document_id": row["document_id"],
        "title": row["title"],
        "content": row["content"],
        "created_at": row["created_at"],
    }
