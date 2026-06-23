import asyncio
import hashlib
from typing import Any

from pycorekit.core_logging.logger import get_logger
from pycorekit.exceptions.file import FileException
from supabase import Client

from app.core.auth import AuthUser
from app.core.cache import cache_get, cache_set, check_rate_limit
from app.core.exceptions import NotFoundException, RateLimitException
from app.services.workspace_access import get_accessible_document, require_editable_document
from app.core.yaml_config import get_yaml_config
from app.services.citations import build_source_citations, collapse_sources_by_document, hash_document_ids
from app.services.embeddings import embed_text, search_workspace_chunks
from app.services.llm_client import (
    answer_multi_document_question,
    generate_document_relevance_summary,
    multi_chat_cache_key,
    semantic_multi_chat_index_key,
)
from app.services.semantic_cache import (
    get_semantic_cached_answer_by_index,
    store_semantic_cached_answer_by_index,
)

log = get_logger("multi_doc")


def _validate_owned_documents(
    client: Client,
    document_ids: list[str],
    user: AuthUser,
    *,
    workspace_id: str | None = None,
) -> list[dict[str, Any]]:
    if not document_ids:
        raise FileException("At least one document is required")

    cfg = get_yaml_config().multi_doc
    unique_ids = list(dict.fromkeys(document_ids))
    if len(unique_ids) > cfg.max_documents:
        raise FileException(f"At most {cfg.max_documents} documents allowed per question")

    rows: list[dict[str, Any]] = []
    for document_id in unique_ids:
        doc = get_accessible_document(client, document_id, user, min_role="viewer")
        rows.append(
            {
                "id": doc["id"],
                "filename": doc["filename"],
                "file_type": doc["file_type"],
                "status": doc["status"],
                "summary": doc.get("summary"),
                "workspace_id": doc.get("workspace_id"),
            }
        )

    if workspace_id:
        for row in rows:
            if row.get("workspace_id") != workspace_id:
                raise FileException("All documents must belong to the selected study set")

    for row in rows:
        if row["file_type"] != "document":
            raise FileException("Multi-doc chat supports document uploads only")
        if row["status"] != "ready":
            raise FileException(
                f"Document {row['filename']} is not processed yet",
                status_code=409,
            )

    return rows


def _similarity_score(row: dict[str, Any]) -> float:
    value = row.get("similarity")
    return float(value) if value is not None else 0.0


def _diversify_context_rows(
    rows: list[dict[str, Any]],
    *,
    limit: int,
    max_per_document: int,
) -> list[dict[str, Any]]:
    """Spread results across documents so one large file does not dominate HITL review."""
    if not rows or limit <= 0:
        return []

    per_doc_cap = max(1, max_per_document)
    sorted_rows = sorted(rows, key=_similarity_score, reverse=True)
    per_doc: dict[str, int] = {}
    selected: list[dict[str, Any]] = []

    for row in sorted_rows:
        doc_id = row["document_id"]
        if per_doc.get(doc_id, 0) >= per_doc_cap:
            continue
        selected.append(row)
        per_doc[doc_id] = per_doc.get(doc_id, 0) + 1
        if len(selected) >= limit:
            break

    return selected


def _retrieve_candidate_limit(cfg: Any, document_count: int) -> int:
    """Fetch extra candidates so per-document caps can still fill the review list."""
    per_doc_cap = max(1, cfg.max_chunks_per_document)
    return min(cfg.max_context_chunks * max(document_count, 1), per_doc_cap * document_count * 4)


def _retrieve_context_rows(
    client: Client,
    *,
    document_ids: list[str],
    question: str,
    doc_map: dict[str, str],
) -> tuple[list[dict[str, Any]], str]:
    cfg = get_yaml_config().multi_doc
    sorted_ids = sorted(document_ids)
    candidate_limit = _retrieve_candidate_limit(cfg, len(sorted_ids))

    query_embedding = embed_text(question)
    matches = search_workspace_chunks(
        client,
        document_ids=sorted_ids,
        query_embedding=query_embedding,
        limit=candidate_limit,
        threshold=get_yaml_config().embeddings.similarity_threshold,
    )
    retrieval_method = "vector"
    if not matches:
        matches = _select_chunks_keyword(
            client,
            document_ids=sorted_ids,
            question=question,
            limit=candidate_limit,
        )
        retrieval_method = "keyword"

    raw_rows = [
        {
            "document_id": row["document_id"],
            "filename": doc_map[row["document_id"]],
            "chunk_index": row["chunk_index"],
            "content": row["content"],
            "similarity": float(row.get("similarity", 0)) if row.get("similarity") is not None else None,
        }
        for row in matches
    ]
    rows = _diversify_context_rows(
        raw_rows,
        limit=cfg.max_context_chunks,
        max_per_document=cfg.max_chunks_per_document,
    )
    return rows, retrieval_method


def _select_chunks_keyword(
    client: Client,
    *,
    document_ids: list[str],
    question: str,
    limit: int,
) -> list[dict[str, Any]]:
    chunks_result = (
        client.table("document_chunks")
        .select("document_id, chunk_index, content")
        .in_("document_id", document_ids)
        .order("chunk_index")
        .execute()
    )
    chunks = chunks_result.data or []
    terms = {term.lower() for term in question.split() if len(term) > 2}
    scored: list[tuple[int, dict[str, Any]]] = []
    for row in chunks:
        content = row["content"]
        lower = content.lower()
        score = sum(lower.count(term) for term in terms)
        scored.append((score, row))
    scored.sort(key=lambda item: item[0], reverse=True)
    selected = [row for score, row in scored if score > 0][:limit]
    if selected:
        return selected
    return chunks[:limit]


def _fallback_document_summary(doc: dict[str, Any]) -> str:
    stored = (doc.get("summary") or "").strip()
    if stored:
        return stored[:1200]
    return "No summary is available for this document yet."


async def _build_document_review_option(
    client: Client,
    *,
    doc: dict[str, Any],
    question: str,
) -> dict[str, Any]:
    doc_id = doc["id"]
    filename = doc["filename"]
    rows, _ = _retrieve_context_rows(
        client,
        document_ids=[doc_id],
        question=question,
        doc_map={doc_id: filename},
    )
    if rows:
        summary = await generate_document_relevance_summary(
            question=question,
            context_chunks=[row["content"] for row in rows],
            filename=filename,
        )
    else:
        summary = _fallback_document_summary(doc)

    return {
        "document_id": doc_id,
        "filename": filename,
        "summary": summary.strip(),
        "selected": True,
    }


def _retrieve_rows_for_documents(
    client: Client,
    *,
    document_ids: list[str],
    question: str,
    doc_map: dict[str, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for document_id in document_ids:
        doc_rows, _ = _retrieve_context_rows(
            client,
            document_ids=[document_id],
            question=question,
            doc_map=doc_map,
        )
        rows.extend(doc_rows)
    return rows


async def retrieve_multiple_documents(
    client: Client,
    user: AuthUser,
    *,
    document_ids: list[str],
    question: str,
    workspace_id: str | None = None,
) -> dict[str, Any]:
    question = question.strip()
    if not question:
        raise FileException("Question is required")

    cfg = get_yaml_config().multi_doc
    allowed, retry_after = await check_rate_limit(
        key=f"multi_retrieve:{user.id}",
        limit=cfg.chat_rate_limit_per_min,
        window_seconds=60,
    )
    if not allowed:
        raise RateLimitException(
            f"Multi-doc retrieve rate limit reached ({cfg.chat_rate_limit_per_min}/min)",
            retry_after=retry_after,
        )

    docs = _validate_owned_documents(client, document_ids, user, workspace_id=workspace_id)
    doc_map = {row["id"]: row["filename"] for row in docs}
    sorted_ids = sorted(doc_map.keys())

    if len(sorted_ids) == 1:
        raise FileException(
            "Document review is only needed when multiple documents are selected",
            status_code=400,
        )

    review_options = await asyncio.gather(
        *[
            _build_document_review_option(client, doc=doc, question=question)
            for doc in sorted(docs, key=lambda row: row["filename"].lower())
        ]
    )

    log.info(
        "Multi-doc document review prepared",
        user_id=user.id,
        document_count=len(sorted_ids),
    )
    return {
        "document_ids": sorted_ids,
        "question": question,
        "documents": list(review_options),
        "hitl_required": True,
        "workspace_id": workspace_id,
    }


async def ask_multiple_documents(
    client: Client,
    user: AuthUser,
    *,
    document_ids: list[str],
    question: str,
    approved_document_ids: list[str],
    workspace_id: str | None = None,
) -> dict[str, Any]:
    question = question.strip()
    if not question:
        raise FileException("Question is required")
    if not approved_document_ids:
        raise FileException("Select at least one document to continue", status_code=400)

    cfg = get_yaml_config().multi_doc
    allowed, retry_after = await check_rate_limit(
        key=f"multi_chat:{user.id}",
        limit=cfg.chat_rate_limit_per_min,
        window_seconds=60,
    )
    if not allowed:
        raise RateLimitException(
            f"Multi-doc chat rate limit reached ({cfg.chat_rate_limit_per_min}/min)",
            retry_after=retry_after,
        )

    docs = _validate_owned_documents(client, document_ids, user, workspace_id=workspace_id)
    doc_map = {row["id"]: row["filename"] for row in docs}
    sorted_ids = sorted(doc_map.keys())
    allowed_ids = set(sorted_ids)

    unique_approved = list(dict.fromkeys(approved_document_ids))
    for document_id in unique_approved:
        if document_id not in allowed_ids:
            raise FileException("One or more selected documents are not in the original selection", status_code=400)

    approved_sorted = sorted(unique_approved)
    docs_hash = hash_document_ids(approved_sorted)
    cache_key = multi_chat_cache_key(user.id, sorted_ids, question, docs_hash)
    cached = await cache_get(cache_key)
    if cached:
        return {**cached, "cached": True}

    semantic_index_key = semantic_multi_chat_index_key(user.id, docs_hash)
    semantic_cached = await get_semantic_cached_answer_by_index(
        index_key=semantic_index_key,
        question=question,
    )
    if semantic_cached:
        _validate_owned_documents(client, approved_sorted, user, workspace_id=workspace_id)
        return semantic_cached

    context_rows = _retrieve_rows_for_documents(
        client,
        document_ids=approved_sorted,
        question=question,
        doc_map=doc_map,
    )
    if not context_rows:
        raise FileException("No document passages found for the selected documents", status_code=409)

    sources = collapse_sources_by_document(build_source_citations(context_rows))

    llm_result = await answer_multi_document_question(
        question=question,
        context_chunks=context_rows,
    )
    payload = {
        "document_ids": approved_sorted,
        "question": question,
        "answer": llm_result["answer"],
        "sources": sources,
        "cached": False,
    }
    await cache_set(cache_key, payload, get_yaml_config().cache.chat_ttl)
    await store_semantic_cached_answer_by_index(
        index_key=semantic_index_key,
        question=question,
        payload=payload,
    )
    log.info(
        "Multi-doc chat answered after document selection",
        user_id=user.id,
        document_count=len(approved_sorted),
        source_count=len(sources),
    )
    return payload


def multi_doc_ids_hash(document_ids: list[str]) -> str:
    joined = ",".join(sorted(document_ids))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]
