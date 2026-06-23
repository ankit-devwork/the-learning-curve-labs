"""Invalidate cached answers when workspace access changes."""

from __future__ import annotations

from supabase import Client

from app.core.cache import cache_delete, cache_delete_pattern
from app.services.llm_client import semantic_chat_index_key, semantic_excel_chat_index_key


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
        document_id = doc["id"]
        await cache_delete(semantic_chat_index_key(user_id, document_id))
        await cache_delete(
            semantic_excel_chat_index_key(user_id, document_id, doc.get("file_hash")),
        )
    await cache_delete_pattern(f"semantic_multi_chat:{user_id}:*")
