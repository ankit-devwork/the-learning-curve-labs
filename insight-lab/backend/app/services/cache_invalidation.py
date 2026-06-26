"""Invalidate cached answers when workspace access or documents change."""

from __future__ import annotations

from supabase import Client

from app.core.cache import cache_delete, cache_delete_pattern
from app.services.llm_client import (
    flashcard_cache_key,
    graph_cache_key,
    infographic_cache_key,
    semantic_chat_index_key,
    semantic_excel_chat_index_key,
    slide_deck_cache_key,
    study_guide_cache_key,
    suggested_questions_cache_key,
    summary_cache_key,
)


async def invalidate_user_workspace_access_caches(
    client: Client,
    *,
    user_id: str,
    workspace_id: str,
) -> None:
    """Drop semantic chat indexes for a user across all documents in a workspace."""
    docs = (
        client.table("documents")
        .select("id, file_hash")
        .eq("workspace_id", workspace_id)
        .execute()
        .data
        or []
    )
    for doc in docs:
        await invalidate_document_caches(
            client,
            user_id=user_id,
            document_id=doc["id"],
            file_hash=doc.get("file_hash"),
        )
    await cache_delete_pattern(f"semantic_multi_chat:{user_id}:*")


async def invalidate_document_caches(
    client: Client,
    *,
    user_id: str,
    document_id: str,
    file_hash: str | None = None,
) -> None:
    """Drop per-document caches for a user after delete or major content change."""
    await cache_delete(summary_cache_key(user_id, document_id))
    await cache_delete(graph_cache_key(user_id, document_id))
    await cache_delete(study_guide_cache_key(user_id, document_id))
    await cache_delete(infographic_cache_key(user_id, document_id))
    await cache_delete(slide_deck_cache_key(user_id, document_id))
    await cache_delete(suggested_questions_cache_key(user_id, document_id))
    await cache_delete(semantic_chat_index_key(user_id, document_id))
    await cache_delete(semantic_excel_chat_index_key(user_id, document_id, file_hash))

    for num_cards in (10, 15, 20, 25, 30):
        await cache_delete(flashcard_cache_key(user_id, document_id, num_cards))

    await cache_delete_pattern(f"chat:{user_id}:{document_id}:*")
    await cache_delete_pattern(f"excel:{user_id}:{document_id}:*")
    await cache_delete_pattern(f"excel_chat:{user_id}:{document_id}:*")
    await cache_delete_pattern(f"quiz:{user_id}:{document_id}:*")
    await cache_delete_pattern(f"excel_quiz:{user_id}:{document_id}:*")
    await cache_delete_pattern(f"adaptive_quiz:{user_id}:{document_id}:*")
    await cache_delete_pattern(f"audio_overview:{user_id}:{document_id}:*")
    await cache_delete_pattern(f"semantic_multi_chat:{user_id}:*")
