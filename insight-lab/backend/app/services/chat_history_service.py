"""Persist and load document / compare chat history."""

from __future__ import annotations

from typing import Any

from supabase import Client

from app.core.auth import AuthUser
from app.services.workspace_access import get_accessible_document, require_workspace_role


def save_document_chat_message(
    client: Client,
    *,
    document_id: str,
    workspace_id: str,
    user: AuthUser,
    question: str,
    answer: str,
    sources: list[dict[str, Any]],
    retrieval_method: str | None = None,
    cached: bool = False,
) -> dict[str, Any]:
    row = (
        client.table("document_chat_messages")
        .insert(
            {
                "document_id": document_id,
                "workspace_id": workspace_id,
                "user_id": user.id,
                "question": question[:2000],
                "answer": answer[:16000],
                "sources": sources,
                "retrieval_method": retrieval_method,
                "cached": cached,
            }
        )
        .execute()
        .data
        or []
    )
    return row[0] if row else {}


def list_document_chat_messages(
    client: Client,
    document_id: str,
    user: AuthUser,
    *,
    limit: int = 100,
) -> list[dict[str, Any]]:
    get_accessible_document(client, document_id, user, min_role="viewer")
    result = (
        client.table("document_chat_messages")
        .select(
            "id, question, answer, sources, retrieval_method, cached, created_at"
        )
        .eq("document_id", document_id)
        .eq("user_id", user.id)
        .order("created_at", desc=False)
        .limit(min(limit, 200))
        .execute()
    )
    return result.data or []


def save_compare_chat_message(
    client: Client,
    *,
    workspace_id: str,
    user: AuthUser,
    document_ids: list[str],
    question: str,
    answer: str,
    sources: list[dict[str, Any]],
    cached: bool = False,
) -> dict[str, Any]:
    require_workspace_role(client, workspace_id, user, min_role="viewer")
    row = (
        client.table("workspace_compare_chat_messages")
        .insert(
            {
                "workspace_id": workspace_id,
                "user_id": user.id,
                "document_ids": document_ids,
                "question": question[:2000],
                "answer": answer[:16000],
                "sources": sources,
                "cached": cached,
            }
        )
        .execute()
        .data
        or []
    )
    return row[0] if row else {}


def list_compare_chat_messages(
    client: Client,
    workspace_id: str,
    user: AuthUser,
    *,
    limit: int = 100,
) -> list[dict[str, Any]]:
    require_workspace_role(client, workspace_id, user, min_role="viewer")
    result = (
        client.table("workspace_compare_chat_messages")
        .select("id, document_ids, question, answer, sources, cached, created_at")
        .eq("workspace_id", workspace_id)
        .eq("user_id", user.id)
        .order("created_at", desc=False)
        .limit(min(limit, 200))
        .execute()
    )
    return result.data or []
