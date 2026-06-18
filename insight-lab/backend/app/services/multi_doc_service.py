import hashlib
from typing import Any

from pycorekit.core_logging.logger import get_logger
from pycorekit.exceptions.file import FileException
from supabase import Client

from app.core.auth import AuthUser
from app.core.cache import cache_get, cache_set, check_rate_limit
from app.core.exceptions import NotFoundException, RateLimitException
from app.core.yaml_config import get_yaml_config
from app.services.citations import build_source_citations, hash_source_refs, source_ref_key
from app.services.embeddings import embed_text, search_workspace_chunks
from app.services.llm_client import answer_multi_document_question, multi_chat_cache_key

log = get_logger("multi_doc")


def _validate_owned_documents(
    client: Client,
    document_ids: list[str],
    user: AuthUser,
) -> list[dict[str, Any]]:
    if not document_ids:
        raise FileException("At least one document is required")

    cfg = get_yaml_config().multi_doc
    unique_ids = list(dict.fromkeys(document_ids))
    if len(unique_ids) > cfg.max_documents:
        raise FileException(f"At most {cfg.max_documents} documents allowed per question")

    result = (
        client.table("documents")
        .select("id, filename, file_type, status")
        .in_("id", unique_ids)
        .eq("owner_id", user.id)
        .execute()
    )
    rows = result.data or []
    if len(rows) != len(unique_ids):
        raise NotFoundException("One or more documents were not found")

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


def _load_approved_rows(
    client: Client,
    *,
    approved_sources: list[dict[str, Any]],
    allowed_document_ids: set[str],
) -> list[dict[str, Any]]:
    if not approved_sources:
        raise FileException("Select at least one source passage to generate an answer", status_code=400)

    rows: list[dict[str, Any]] = []
    for ref in approved_sources:
        document_id = ref["document_id"]
        chunk_index = ref["chunk_index"]
        if document_id not in allowed_document_ids:
            raise FileException("One or more approved sources are not in the selected documents", status_code=400)

        result = (
            client.table("document_chunks")
            .select("document_id, chunk_index, content")
            .eq("document_id", document_id)
            .eq("chunk_index", chunk_index)
            .limit(1)
            .execute()
        )
        if not result.data:
            raise FileException(
                f"Source passage not found for document {document_id}",
                status_code=400,
            )
        rows.append(result.data[0])

    doc_result = (
        client.table("documents")
        .select("id, filename")
        .in_("id", list({row["document_id"] for row in rows}))
        .execute()
    )
    doc_map = {row["id"]: row["filename"] for row in (doc_result.data or [])}

    return [
        {
            "document_id": row["document_id"],
            "filename": doc_map.get(row["document_id"], "Document"),
            "chunk_index": row["chunk_index"],
            "content": row["content"],
            "similarity": ref.get("similarity"),
        }
        for row, ref in zip(rows, approved_sources, strict=True)
    ]


async def retrieve_multiple_documents(
    client: Client,
    user: AuthUser,
    *,
    document_ids: list[str],
    question: str,
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

    docs = _validate_owned_documents(client, document_ids, user)
    doc_map = {row["id"]: row["filename"] for row in docs}
    sorted_ids = sorted(doc_map.keys())

    rows, retrieval_method = _retrieve_context_rows(
        client,
        document_ids=sorted_ids,
        question=question,
        doc_map=doc_map,
    )
    if not rows:
        raise FileException("No document passages found for the selected documents", status_code=409)

    sources = build_source_citations(rows)
    for source in sources:
        source["selected"] = True

    log.info(
        "Multi-doc sources retrieved for review",
        user_id=user.id,
        document_count=len(sorted_ids),
        source_count=len(sources),
    )
    return {
        "document_ids": sorted_ids,
        "question": question,
        "sources": sources,
        "retrieval_method": retrieval_method,
        "hitl_required": True,
    }


async def ask_multiple_documents(
    client: Client,
    user: AuthUser,
    *,
    document_ids: list[str],
    question: str,
    approved_sources: list[dict[str, Any]],
) -> dict[str, Any]:
    question = question.strip()
    if not question:
        raise FileException("Question is required")

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

    docs = _validate_owned_documents(client, document_ids, user)
    doc_map = {row["id"]: row["filename"] for row in docs}
    sorted_ids = sorted(doc_map.keys())
    allowed_ids = set(sorted_ids)

    refs_hash = hash_source_refs(approved_sources)
    cache_key = multi_chat_cache_key(user.id, sorted_ids, question, refs_hash)
    cached = await cache_get(cache_key)
    if cached:
        return {**cached, "cached": True}

    context_rows = _load_approved_rows(
        client,
        approved_sources=approved_sources,
        allowed_document_ids=allowed_ids,
    )
    sources = build_source_citations(context_rows)

    llm_result = await answer_multi_document_question(
        question=question,
        context_chunks=context_rows,
    )
    payload = {
        "document_ids": sorted_ids,
        "question": question,
        "answer": llm_result["answer"],
        "sources": sources,
        "cached": False,
    }
    await cache_set(cache_key, payload, get_yaml_config().cache.chat_ttl)
    log.info(
        "Multi-doc chat answered after HITL approval",
        user_id=user.id,
        document_count=len(sorted_ids),
        source_count=len(sources),
    )
    return payload


def multi_doc_ids_hash(document_ids: list[str]) -> str:
    joined = ",".join(sorted(document_ids))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]
