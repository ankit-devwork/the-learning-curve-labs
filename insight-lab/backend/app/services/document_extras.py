from supabase import Client

from app.core.auth import AuthUser
from app.core.cache import cache_get, cache_set
from app.core.exceptions import NotFoundException
from app.core.safe_errors import sanitize_stored_error
from app.core.yaml_config import get_yaml_config
from app.services.llm_client import suggested_questions_cache_key
from app.services.quiz_service import _get_owned_document

DEFAULT_SUGGESTED_QUESTIONS = [
    "What are the main themes in this document?",
    "Summarize the key conclusions in plain language.",
    "What definitions or terms should I remember?",
    "How would you explain this to someone new to the topic?",
]


async def get_document_processing_status(client: Client, document_id: str, user: AuthUser) -> dict:
    doc = _get_owned_document(client, document_id, user)
    status = doc["status"]
    if status == "pending":
        return {
            "document_id": document_id,
            "status": status,
            "stage": "queued",
            "progress_pct": 10,
            "message": "Waiting to process…",
        }
    if status == "processing":
        return {
            "document_id": document_id,
            "status": status,
            "stage": "processing",
            "progress_pct": 55,
            "message": "Extracting text and building search embeddings…",
        }
    if status == "ready":
        return {
            "document_id": document_id,
            "status": status,
            "stage": "ready",
            "progress_pct": 100,
            "message": "Ready — you can ask questions, take a quiz, or generate flashcards.",
        }
    return {
        "document_id": document_id,
        "status": status,
        "stage": "failed",
        "progress_pct": 0,
        "message": sanitize_stored_error(doc.get("error_message")) or "Processing failed.",
    }


async def get_document_chunk(
    client: Client,
    document_id: str,
    chunk_index: int,
    user: AuthUser,
) -> dict:
    _get_owned_document(client, document_id, user)
    if chunk_index < 0 or chunk_index > 10_000:
        raise NotFoundException("Chunk not found")
    result = (
        client.table("document_chunks")
        .select("chunk_index, content")
        .eq("document_id", document_id)
        .eq("chunk_index", chunk_index)
        .limit(1)
        .execute()
    )
    if not result.data:
        raise NotFoundException("Chunk not found")
    row = result.data[0]
    preview_limit = get_yaml_config().artifacts.chunk_preview_max_chars
    content = row["content"]
    return {
        "document_id": document_id,
        "chunk_index": row["chunk_index"],
        "content": content,
        "preview": content[:preview_limit],
        "truncated": len(content) > preview_limit,
    }


async def get_suggested_questions(client: Client, document_id: str, user: AuthUser) -> dict:
    doc = _get_owned_document(client, document_id, user)
    if doc["file_type"] != "document":
        return {"document_id": document_id, "questions": DEFAULT_SUGGESTED_QUESTIONS[:3]}
    if doc["status"] != "ready":
        return {"document_id": document_id, "questions": DEFAULT_SUGGESTED_QUESTIONS[:3]}

    cache_key = suggested_questions_cache_key(user.id, document_id)
    cached = await cache_get(cache_key)
    if cached and cached.get("questions"):
        return {"document_id": document_id, "questions": cached["questions"], "cached": True}

    summary = (doc.get("summary") or "").strip()
    if not summary:
        return {"document_id": document_id, "questions": DEFAULT_SUGGESTED_QUESTIONS}

    from app.services.llm_client import generate_suggested_questions

    questions = await generate_suggested_questions(summary=summary, filename=doc["filename"])
    payload = {"questions": questions[:6]}
    await cache_set(cache_key, payload, get_yaml_config().cache.artifact_ttl)
    return {"document_id": document_id, "questions": payload["questions"], "cached": False}


def attach_source_previews(
    client: Client,
    *,
    document_id: str,
    results: list[dict],
    questions: list[dict],
) -> list[dict]:
    preview_limit = get_yaml_config().artifacts.chunk_preview_max_chars
    chunk_cache: dict[int, str] = {}
    question_by_id = {row["id"]: row for row in questions}
    enriched: list[dict] = []
    for result in results:
        item = dict(result)
        if result.get("correct"):
            enriched.append(item)
            continue
        question = question_by_id.get(result["question_id"], {})
        source_chunk_id = question.get("source_chunk_id")
        if source_chunk_id is None:
            enriched.append(item)
            continue
        try:
            chunk_index = int(source_chunk_id)
        except (TypeError, ValueError):
            enriched.append(item)
            continue
        if chunk_index not in chunk_cache:
            chunk = (
                client.table("document_chunks")
                .select("content")
                .eq("document_id", document_id)
                .eq("chunk_index", chunk_index)
                .limit(1)
                .execute()
            )
            if chunk.data:
                content = chunk.data[0]["content"]
                chunk_cache[chunk_index] = content[:preview_limit]
            else:
                chunk_cache[chunk_index] = ""
        item["source_preview"] = chunk_cache.get(chunk_index) or None
        item["source_chunk_index"] = chunk_index
        enriched.append(item)
    return enriched
