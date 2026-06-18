import hashlib
from datetime import datetime, timezone

from pycorekit.core_logging.logger import get_logger
from pycorekit.exceptions.base import AppException
from pycorekit.exceptions.file import FileException
from supabase import Client

from app.core.auth import AuthUser
from app.core.cache import cache_get, cache_set, check_rate_limit
from app.core.exceptions import NotFoundException, RateLimitException
from app.core.yaml_config import get_yaml_config
from app.services.chunking import chunk_text
from app.services.document_text import extract_text_from_bytes
from app.services.embeddings import (
    embed_text,
    embed_texts_batched,
    search_similar_chunks,
)
from app.core.safe_errors import GENERIC_PROCESSING_ERROR, sanitize_stored_error
from app.services.graph_service import sync_document_graph
from app.services.citations import build_source_citations
from app.services.llm_client import (
    answer_question,
    generate_summary,
    question_cache_key,
    summary_cache_key,
)

log = get_logger("documents")


def _get_owned_document(client: Client, document_id: str, user: AuthUser) -> dict:
    result = (
        client.table("documents")
        .select("*")
        .eq("id", document_id)
        .eq("owner_id", user.id)
        .limit(1)
        .execute()
    )
    if not result.data:
        raise NotFoundException("Document not found")
    return result.data[0]


def _download_document_bytes(client: Client, document: dict) -> bytes:
    upload = get_yaml_config().upload
    try:
        return client.storage.from_(upload.storage_bucket).download(document["storage_path"])
    except Exception as exc:
        raise FileException(f"Failed to download document from storage: {exc}", status_code=502) from exc


async def get_document(client: Client, document_id: str, user: AuthUser) -> dict:
    doc = _get_owned_document(client, document_id, user)
    return {
        "id": doc["id"],
        "filename": doc["filename"],
        "file_type": doc["file_type"],
        "mime_type": doc.get("mime_type"),
        "status": doc["status"],
        "summary": doc.get("summary"),
        "error_message": sanitize_stored_error(doc.get("error_message")),
        "created_at": doc["created_at"],
        "processed_at": doc.get("processed_at"),
    }


async def get_document_summary(client: Client, document_id: str, user: AuthUser) -> dict:
    doc = _get_owned_document(client, document_id, user)
    if doc["file_type"] != "document":
        raise FileException("Summaries are only available for document uploads")

    cache_cfg = get_yaml_config().cache
    cache_key = summary_cache_key(user.id, document_id)
    cached = await cache_get(cache_key)
    if cached:
        return {**cached, "cached": True}

    if doc["status"] != "ready" or not doc.get("summary"):
        raise FileException("Document is not processed yet", status_code=409)

    payload = {
        "document_id": document_id,
        "summary": doc["summary"],
        "status": doc["status"],
        "cached": False,
    }
    await cache_set(cache_key, payload, cache_cfg.summary_ttl)
    return payload


async def process_document(client: Client, document_id: str, user: AuthUser) -> dict:
    cfg = get_yaml_config().documents
    allowed, retry_after = await check_rate_limit(
        key=f"process:{user.id}",
        limit=cfg.process_rate_limit_per_hour,
        window_seconds=3600,
    )
    if not allowed:
        raise RateLimitException(
            f"Document processing limit reached ({cfg.process_rate_limit_per_hour}/hour)",
            retry_after=retry_after,
        )

    doc = _get_owned_document(client, document_id, user)
    if doc["file_type"] != "document":
        raise FileException("Only document uploads can be processed in Step 1.7")
    if doc["status"] == "processing":
        raise FileException("Document is already processing", status_code=409)

    client.table("documents").update(
        {"status": "processing", "error_message": None}
    ).eq("id", document_id).execute()

    try:
        raw_bytes = _download_document_bytes(client, doc)
        text = extract_text_from_bytes(raw_bytes, doc["filename"])
        chunks = chunk_text(
            text,
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
        )
        if not chunks:
            raise FileException("No text content found after extraction")

        client.table("document_chunks").delete().eq("document_id", document_id).execute()
        vectors = embed_texts_batched(chunks)
        if len(vectors) != len(chunks):
            raise FileException(
                f"Embedding count mismatch: {len(vectors)} vectors for {len(chunks)} chunks",
                status_code=500,
            )
        chunk_rows = [
            {
                "document_id": document_id,
                "chunk_index": index,
                "content": chunk,
                "token_count": len(chunk.split()),
                # PostgREST expects a JSON array of floats for pgvector columns.
                "embedding": vectors[index],
            }
            for index, chunk in enumerate(chunks)
        ]
        client.table("document_chunks").insert(chunk_rows).execute()

        summary = await generate_summary(text, filename=doc["filename"])
        updated = (
            client.table("documents")
            .update(
                {
                    "status": "ready",
                    "summary": summary,
                    "error_message": None,
                    "processed_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            .eq("id", document_id)
            .execute()
        )
        row = updated.data[0] if updated.data else doc
        cache_key = summary_cache_key(user.id, document_id)
        await cache_set(
            cache_key,
            {
                "document_id": document_id,
                "summary": summary,
                "status": "ready",
            },
            get_yaml_config().cache.summary_ttl,
        )
        log.info(
            "Document processed",
            document_id=document_id,
            user_id=user.id,
            chunk_count=len(chunks),
            embedded_count=len(vectors),
        )
        try:
            await sync_document_graph(
                client,
                document_id,
                user,
                skip_rate_limit=True,
            )
        except Exception as exc:
            log.warning(
                "Concept graph sync skipped after processing",
                document_id=document_id,
                error=str(exc),
            )
        return {
            "document_id": document_id,
            "status": row.get("status", "ready"),
            "chunk_count": len(chunks),
            "embedded_count": len(vectors),
            "summary_preview": summary[:280],
        }
    except AppException as exc:
        client.table("documents").update(
            {"status": "failed", "error_message": exc.message}
        ).eq("id", document_id).execute()
        raise
    except Exception as exc:
        log.exception("Document processing failed", document_id=document_id, user_id=user.id)
        client.table("documents").update(
            {"status": "failed", "error_message": GENERIC_PROCESSING_ERROR}
        ).eq("id", document_id).execute()
        raise FileException(GENERIC_PROCESSING_ERROR, status_code=500) from exc


def _select_relevant_chunks_keyword(chunks: list[dict], question: str, *, limit: int) -> list[dict]:
    terms = {term.lower() for term in question.split() if len(term) > 2}
    scored: list[tuple[int, dict]] = []
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


def _select_relevant_chunks_vector(
    client: Client,
    *,
    document_id: str,
    question: str,
    limit: int,
) -> tuple[list[dict], str]:
    cfg = get_yaml_config().embeddings
    query_vector = embed_text(question)
    matches = search_similar_chunks(
        client,
        document_id=document_id,
        query_embedding=query_vector,
        limit=limit,
        threshold=cfg.similarity_threshold,
    )
    if matches:
        rows = [
            {
                "chunk_index": row["chunk_index"],
                "content": row["content"],
                "similarity": float(row.get("similarity", 0)),
            }
            for row in matches
        ]
        return rows, "vector"
    return [], "vector"


async def ask_document(
    client: Client,
    document_id: str,
    user: AuthUser,
    *,
    question: str,
) -> dict:
    question = question.strip()
    if not question:
        raise FileException("Question is required")

    cfg = get_yaml_config().documents
    allowed, retry_after = await check_rate_limit(
        key=f"chat:{user.id}",
        limit=cfg.chat_rate_limit_per_min,
        window_seconds=60,
    )
    if not allowed:
        raise RateLimitException(
            f"Chat rate limit reached ({cfg.chat_rate_limit_per_min}/min)",
            retry_after=retry_after,
        )

    doc = _get_owned_document(client, document_id, user)
    if doc["file_type"] != "document":
        raise FileException("Chat is only available for document uploads")
    if doc["status"] != "ready":
        raise FileException("Document must be processed before asking questions", status_code=409)

    cache_key = question_cache_key(user.id, document_id, question)
    cached = await cache_get(cache_key)
    if cached:
        return {**cached, "cached": True}

    chunks_result = (
        client.table("document_chunks")
        .select("chunk_index, content")
        .eq("document_id", document_id)
        .order("chunk_index")
        .execute()
    )
    chunks = chunks_result.data or []
    if not chunks:
        raise FileException("No document chunks found; reprocess the document", status_code=409)

    limit = get_yaml_config().documents.max_context_chunks
    selected_rows, retrieval_method = _select_relevant_chunks_vector(
        client,
        document_id=document_id,
        question=question,
        limit=limit,
    )
    if not selected_rows:
        selected_rows = _select_relevant_chunks_keyword(chunks, question, limit=limit)
        retrieval_method = "keyword"

    sources = build_source_citations(
        selected_rows,
        filename=doc["filename"],
        document_id=document_id,
    )
    context_chunks = [row["content"] for row in selected_rows]

    llm_result = await answer_question(
        question=question,
        context_chunks=context_chunks,
        filename=doc["filename"],
    )
    payload = {
        "document_id": document_id,
        "question": question,
        "answer": llm_result["answer"],
        "sources": sources,
        "retrieval_method": retrieval_method,
        "cached": False,
    }
    await cache_set(cache_key, payload, get_yaml_config().cache.chat_ttl)
    return payload
