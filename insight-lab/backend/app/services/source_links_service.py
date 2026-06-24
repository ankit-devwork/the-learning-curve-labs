"""Link Excel spreadsheets to related documents in the same study sheet."""

from __future__ import annotations

import secrets
from typing import Any
from uuid import uuid4

from pycorekit.exceptions.file import FileException
from supabase import Client

from app.core.auth import AuthUser
from app.services.citations import build_source_citations, collapse_sources_by_document
from app.services.document_service import _select_relevant_chunks_keyword, _select_relevant_chunks_vector
from app.services.workspace_access import get_accessible_document, require_workspace_role


def list_source_links(client: Client, workspace_id: str, user: AuthUser) -> list[dict[str, Any]]:
    require_workspace_role(client, workspace_id, user, min_role="viewer")
    rows = (
        client.table("document_source_links")
        .select("id, excel_document_id, document_id, label, created_at")
        .eq("workspace_id", workspace_id)
        .order("created_at", desc=True)
        .execute()
        .data
        or []
    )
    if not rows:
        return []

    doc_ids = {row["excel_document_id"] for row in rows} | {row["document_id"] for row in rows}
    docs = (
        client.table("documents")
        .select("id, filename, file_type")
        .in_("id", list(doc_ids))
        .execute()
        .data
        or []
    )
    name_by_id = {row["id"]: row["filename"] for row in docs}
    type_by_id = {row["id"]: row["file_type"] for row in docs}

    return [
        {
            **row,
            "excel_filename": name_by_id.get(row["excel_document_id"]),
            "document_filename": name_by_id.get(row["document_id"]),
            "excel_file_type": type_by_id.get(row["excel_document_id"]),
            "document_file_type": type_by_id.get(row["document_id"]),
        }
        for row in rows
    ]


def create_source_link(
    client: Client,
    workspace_id: str,
    user: AuthUser,
    *,
    excel_document_id: str,
    document_id: str,
    label: str | None = None,
) -> dict[str, Any]:
    require_workspace_role(client, workspace_id, user, min_role="editor")
    excel_doc = get_accessible_document(client, excel_document_id, user, min_role="editor")
    text_doc = get_accessible_document(client, document_id, user, min_role="editor")
    if excel_doc["workspace_id"] != workspace_id or text_doc["workspace_id"] != workspace_id:
        raise FileException("Both files must belong to this study sheet", status_code=400)
    if excel_doc["file_type"] != "excel":
        raise FileException("Link source must be a spreadsheet", status_code=400)
    if text_doc["file_type"] != "document":
        raise FileException("Link target must be a document", status_code=400)

    row = {
        "id": str(uuid4()),
        "workspace_id": workspace_id,
        "excel_document_id": excel_document_id,
        "document_id": document_id,
        "label": (label or "").strip()[:120] or None,
        "created_by": user.id,
    }
    inserted = client.table("document_source_links").insert(row).select("*").execute()
    if not inserted.data:
        raise FileException("Failed to create source link", status_code=500)
    return inserted.data[0]


def delete_source_link(
    client: Client,
    workspace_id: str,
    user: AuthUser,
    *,
    link_id: str,
) -> None:
    require_workspace_role(client, workspace_id, user, min_role="editor")
    client.table("document_source_links").delete().eq("id", link_id).eq(
        "workspace_id", workspace_id
    ).execute()


def fetch_linked_document_chunks(
    client: Client,
    *,
    excel_document_id: str,
    question: str,
    limit: int = 3,
) -> list[dict[str, Any]]:
    links = (
        client.table("document_source_links")
        .select("document_id")
        .eq("excel_document_id", excel_document_id)
        .execute()
        .data
        or []
    )
    if not links:
        return []

    selected: list[dict[str, Any]] = []
    per_doc = max(1, limit // max(len(links), 1))
    for link in links:
        doc_id = link["document_id"]
        doc = (
            client.table("documents")
            .select("id, filename, status")
            .eq("id", doc_id)
            .limit(1)
            .execute()
        )
        if not doc.data or doc.data[0]["status"] != "ready":
            continue
        filename = doc.data[0]["filename"]
        rows, _method = _select_relevant_chunks_vector(
            client,
            document_id=doc_id,
            question=question,
            limit=per_doc,
        )
        if not rows:
            chunks = (
                client.table("document_chunks")
                .select("chunk_index, content")
                .eq("document_id", doc_id)
                .order("chunk_index")
                .execute()
                .data
                or []
            )
            rows = _select_relevant_chunks_keyword(chunks, question, limit=per_doc)
        for row in rows:
            row["document_id"] = doc_id
            row["filename"] = filename
        selected.extend(rows)

    return selected[:limit]


def build_linked_citations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    return collapse_sources_by_document(build_source_citations(rows))


def new_public_share_token() -> str:
    return secrets.token_urlsafe(24)
