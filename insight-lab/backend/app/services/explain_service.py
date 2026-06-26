"""Deep explanations for quiz questions and flashcards with citations."""

from __future__ import annotations

import json
from typing import Any

from pycorekit.exceptions.file import FileException
from supabase import Client

from app.core.auth import AuthUser
from app.core.cache import check_rate_limit
from app.core.exceptions import NotFoundException, RateLimitException
from app.core.yaml_config import get_yaml_config
from app.services.citations import build_source_citations, collapse_sources_by_document
from app.services.llm_client import explain_with_citations
from app.services.quiz_service import _sample_chunks
from app.services.workspace_access import get_accessible_document


async def explain_quiz_question(
    client: Client,
    quiz_id: str,
    question_id: str,
    user: AuthUser,
) -> dict[str, Any]:
    cfg = get_yaml_config().explain
    allowed, retry_after = await check_rate_limit(
        key=f"explain:{user.id}",
        limit=cfg.rate_limit_per_min,
        window_seconds=60,
    )
    if not allowed:
        raise RateLimitException(
            f"Explain limit reached ({cfg.rate_limit_per_min}/min)",
            retry_after=retry_after,
        )

    quiz = (
        client.table("quizzes")
        .select("id, document_id, title")
        .eq("id", quiz_id)
        .limit(1)
        .execute()
        .data
        or []
    )
    if not quiz:
        raise NotFoundException("Quiz not found")
    quiz_row = quiz[0]
    doc = get_accessible_document(client, quiz_row["document_id"], user, min_role="viewer")

    question = (
        client.table("quiz_questions")
        .select("*")
        .eq("id", question_id)
        .eq("quiz_id", quiz_id)
        .limit(1)
        .execute()
        .data
        or []
    )
    if not question:
        raise NotFoundException("Question not found")
    q = question[0]

    chunks = (
        client.table("document_chunks")
        .select("chunk_index, content")
        .eq("document_id", doc["id"])
        .order("chunk_index")
        .execute()
        .data
        or []
    )
    if not chunks:
        raise FileException("Document has no chunks", status_code=409)

    cfg_art = get_yaml_config().artifacts
    sampled = _sample_chunks(chunks, cfg_art.max_context_chunks)
    if q.get("source_chunk_id"):
        try:
            idx = int(str(q["source_chunk_id"]).split("_")[-1])
            match = next((c for c in chunks if c["chunk_index"] == idx), None)
            if match and match not in sampled:
                sampled = [match, *sampled[: cfg_art.max_context_chunks - 1]]
        except (ValueError, TypeError):
            pass

    sources = collapse_sources_by_document(
        build_source_citations(sampled, filename=doc["filename"], document_id=doc["id"])
    )
    context = [row["content"] for row in sampled]
    options = q.get("options") or []
    correct_idx = int(q.get("correct_option_index") or 0)
    correct_text = options[correct_idx] if 0 <= correct_idx < len(options) else ""

    raw = await explain_with_citations(
        topic=q["question_text"],
        context_label=f"Quiz: {quiz_row['title']}",
        context_chunks=context,
        filename=doc["filename"],
        extra=f"Correct answer: {correct_text}. Existing explanation: {q.get('explanation') or 'none'}",
    )
    return {
        "kind": "quiz_question",
        "quiz_id": quiz_id,
        "question_id": question_id,
        "explanation": raw.get("explanation") or "",
        "sources": sources,
    }


async def explain_flashcard(
    client: Client,
    set_id: str,
    flashcard_id: str,
    user: AuthUser,
) -> dict[str, Any]:
    cfg = get_yaml_config().explain
    allowed, retry_after = await check_rate_limit(
        key=f"explain:{user.id}",
        limit=cfg.rate_limit_per_min,
        window_seconds=60,
    )
    if not allowed:
        raise RateLimitException(
            f"Explain limit reached ({cfg.rate_limit_per_min}/min)",
            retry_after=retry_after,
        )

    card = (
        client.table("flashcards")
        .select("id, front, back, source_chunk_index, set_id")
        .eq("id", flashcard_id)
        .eq("set_id", set_id)
        .limit(1)
        .execute()
        .data
        or []
    )
    if not card:
        raise NotFoundException("Flashcard not found")
    c = card[0]

    flashcard_set = (
        client.table("flashcard_sets")
        .select("document_id")
        .eq("id", set_id)
        .limit(1)
        .execute()
        .data
        or []
    )
    if not flashcard_set:
        raise NotFoundException("Flashcard set not found")
    doc = get_accessible_document(client, flashcard_set[0]["document_id"], user, min_role="viewer")

    chunks = (
        client.table("document_chunks")
        .select("chunk_index, content")
        .eq("document_id", doc["id"])
        .order("chunk_index")
        .execute()
        .data
        or []
    )
    if not chunks:
        raise FileException("Document has no chunks", status_code=409)

    selected = chunks
    if c.get("source_chunk_index") is not None:
        idx = c["source_chunk_index"]
        match = next((row for row in chunks if row["chunk_index"] == idx), None)
        selected = [match] if match else _sample_chunks(chunks, 3)
    else:
        selected = _sample_chunks(chunks, 3)

    sources = collapse_sources_by_document(
        build_source_citations(selected, filename=doc["filename"], document_id=doc["id"])
    )
    raw = await explain_with_citations(
        topic=f"{c['front']} → {c['back']}",
        context_label="Flashcard",
        context_chunks=[row["content"] for row in selected],
        filename=doc["filename"],
        extra="Explain the term and definition clearly for a learner who marked this card as difficult.",
    )
    return {
        "kind": "flashcard",
        "set_id": set_id,
        "flashcard_id": flashcard_id,
        "explanation": raw.get("explanation") or "",
        "sources": sources,
    }
