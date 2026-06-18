import hashlib
from typing import Any

from pycorekit.core_logging.logger import get_logger
from pycorekit.exceptions.file import FileException
from supabase import Client

from app.core.auth import AuthUser
from app.core.cache import cache_get, cache_set, check_rate_limit
from app.core.exceptions import NotFoundException, RateLimitException
from app.core.yaml_config import get_yaml_config
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
    return [row for _, row in scored[:limit]]


async def ask_multiple_documents(
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

    cache_key = multi_chat_cache_key(user.id, sorted_ids, question)
    cached = await cache_get(cache_key)
    if cached:
        return {**cached, "cached": True}

    query_embedding = embed_text(question)
    matches = search_workspace_chunks(
        client,
        document_ids=sorted_ids,
        query_embedding=query_embedding,
        limit=cfg.max_context_chunks,
        threshold=get_yaml_config().embeddings.similarity_threshold,
    )
    retrieval_method = "vector"
    if not matches:
        matches = _select_chunks_keyword(
            client,
            document_ids=sorted_ids,
            question=question,
            limit=cfg.max_context_chunks,
        )
        retrieval_method = "keyword"

    if not matches:
        raise FileException("No document chunks found for the selected documents", status_code=409)

    context_chunks = [
        {
            "document_id": row["document_id"],
            "filename": doc_map[row["document_id"]],
            "chunk_index": row["chunk_index"],
            "content": row["content"],
            "similarity": float(row.get("similarity", 0)),
        }
        for row in matches
    ]

    llm_result = await answer_multi_document_question(
        question=question,
        context_chunks=context_chunks,
    )
    payload = {
        "document_ids": sorted_ids,
        "question": question,
        "answer": llm_result["answer"],
        "cited_documents": llm_result["cited_documents"],
        "retrieval_method": retrieval_method,
        "cached": False,
    }
    await cache_set(cache_key, payload, get_yaml_config().cache.chat_ttl)
    log.info(
        "Multi-doc chat answered",
        user_id=user.id,
        document_count=len(sorted_ids),
        retrieval_method=retrieval_method,
    )
    return payload


def multi_doc_ids_hash(document_ids: list[str]) -> str:
    joined = ",".join(sorted(document_ids))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]
