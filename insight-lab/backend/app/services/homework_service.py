"""Step-by-step homework help grounded in uploaded course materials."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from pycorekit.exceptions.file import FileException
from supabase import Client

from app.core.auth import AuthUser
from app.core.cache import check_rate_limit
from app.core.exceptions import RateLimitException
from app.core.yaml_config import get_yaml_config
from app.services.citations import build_source_citations, collapse_sources_by_document
from app.services.document_service import (
    _select_relevant_chunks_keyword,
    _select_relevant_chunks_vector,
)
from app.services.llm_client import solve_homework_step_by_step
from app.services.workspace_access import get_accessible_document


async def solve_document_homework(
    client: Client,
    document_id: str,
    user: AuthUser,
    *,
    question: str,
) -> dict[str, Any]:
    question = question.strip()
    if not question:
        raise FileException("Question is required")
    if len(question) > 4000:
        raise FileException("Question is too long (max 4000 characters)")

    cfg = get_yaml_config().homework
    allowed, retry_after = await check_rate_limit(
        key=f"homework:{user.id}",
        limit=cfg.rate_limit_per_min,
        window_seconds=60,
    )
    if not allowed:
        raise RateLimitException(
            f"Homework solver limit reached ({cfg.rate_limit_per_min}/min)",
            retry_after=retry_after,
        )

    doc = get_accessible_document(client, document_id, user, min_role="viewer")
    if doc["file_type"] != "document":
        raise FileException("Homework solver is only available for document uploads")
    if doc["status"] != "ready":
        raise FileException("Document must be processed first", status_code=409)

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
        raise FileException("No document chunks found", status_code=409)

    limit = get_yaml_config().documents.max_context_chunks
    selected, method = _select_relevant_chunks_vector(
        client, document_id=document_id, question=question, limit=limit
    )
    if not selected:
        selected = _select_relevant_chunks_keyword(chunks, question, limit=limit)
        method = "keyword"

    sources = collapse_sources_by_document(
        build_source_citations(selected, filename=doc["filename"], document_id=document_id)
    )
    context = [row["content"] for row in selected]

    raw = await solve_homework_step_by_step(
        question=question,
        context_chunks=context,
        filename=doc["filename"],
    )

    solution_id = str(uuid4())
    answer_payload = {
        "steps": raw.get("steps") or [],
        "summary": raw.get("summary") or "",
        "sources": sources,
        "retrieval_method": method,
        "disclaimer": (
            "This solution is grounded in your uploaded materials when possible. "
            "Verify critical steps with your instructor or textbook."
        ),
    }
    client.table("document_homework_solutions").insert(
        {
            "id": solution_id,
            "document_id": document_id,
            "workspace_id": doc["workspace_id"],
            "user_id": user.id,
            "question": question,
            "answer": answer_payload,
        }
    ).execute()

    return {
        "solution_id": solution_id,
        "document_id": document_id,
        "question": question,
        **answer_payload,
    }


def list_homework_solutions(
    client: Client,
    document_id: str,
    user: AuthUser,
    *,
    limit: int = 20,
) -> list[dict[str, Any]]:
    get_accessible_document(client, document_id, user, min_role="viewer")
    rows = (
        client.table("document_homework_solutions")
        .select("id, question, answer, created_at")
        .eq("document_id", document_id)
        .eq("user_id", user.id)
        .order("created_at", desc=True)
        .limit(min(limit, 50))
        .execute()
        .data
        or []
    )
    return rows
