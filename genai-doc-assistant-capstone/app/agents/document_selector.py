import numpy as np
import asyncio

from app.service.db_connection import (
    async_get_collection,
    async_embed_texts
)
from app.core.settings import settings

# Unified tracing engine
from pycorekit.tracing.tracing import start_trace


def _flatten_metadatas(raw_metadatas):
    if not raw_metadatas:
        return []
    if isinstance(raw_metadatas, list) and raw_metadatas and isinstance(raw_metadatas[0], list):
        return [item for batch in raw_metadatas for item in batch if isinstance(item, dict)]
    return raw_metadatas

# DB observability wrapper
from pycorekit.observability.observability import observe_db

from pycorekit.correlation.context import get_current_correlation_id


async def document_selector_agent(state):
    cid = get_current_correlation_id() or "unknown"

    with start_trace("document_selector_agent", inputs={"state": dict(state)}):
        question = state["question"]
        margin = settings.rag.document_selection_margin

        # ---------------------------------------------------------
        # 1. Load documents (DB call)
        # ---------------------------------------------------------
        with start_trace("load_documents"):
            # First await the async collection fetch
            collection = await async_get_collection("documents")

            # Then wrap the DB call in observe_db
            docs = await observe_db(
                "chroma_get_documents",
                func=lambda: asyncio.to_thread(
                    collection.get,
                    include=["metadatas"]
                ),
                query="collection.get(include=['metadatas'])"
            )

        metadatas = _flatten_metadatas(docs.get("metadatas", []))
        if not docs or not metadatas:
            return {
                **state,
                "selected_doc_id": None,
                "needs_user_choice": False,
                "candidate_docs": []
            }

        # ---------------------------------------------------------
        # 2. Build unique doc list
        # ---------------------------------------------------------
        with start_trace("build_doc_map"):
            doc_map = {}
            for meta in metadatas:
                doc_id = meta["doc_id"]
                if doc_id not in doc_map:
                    doc_map[doc_id] = {
                        "doc_id": doc_id,
                        "title": meta.get("title", "Untitled Document"),
                        "summary": meta.get("summary", ""),
                        "filename": meta.get("filename") or meta.get("path", "").split("/")[-1] or "unknown",
                    }

            documents = list(doc_map.values())

        # ---------------------------------------------------------
        # 3. Only one document → auto-select
        # ---------------------------------------------------------
        if len(documents) == 1:
            return {
                **state,
                "selected_doc_id": documents[0]["doc_id"],
                "needs_user_choice": False,
                "candidate_docs": documents
            }

        # ---------------------------------------------------------
        # 4. Embed question
        # ---------------------------------------------------------
        with start_trace("embed_question"):
            query_embedding = (await async_embed_texts([question]))[0]

        # ---------------------------------------------------------
        # 5. Batch embed all document summaries
        # ---------------------------------------------------------
        with start_trace("embed_documents"):
            texts = [
                f"{d['title']} {d['summary']}".strip()
                for d in documents
            ]
            doc_embeddings = await async_embed_texts(texts)

        # ---------------------------------------------------------
        # 6. Score documents
        # ---------------------------------------------------------
        with start_trace("score_documents"):
            scored_docs = []
            for doc, doc_emb in zip(documents, doc_embeddings):
                score = float(
                    np.dot(query_embedding, doc_emb)
                    / (np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb))
                )
                scored_docs.append((doc, score))

        # ---------------------------------------------------------
        # 7. Sort
        # ---------------------------------------------------------
        with start_trace("sort_documents"):
            scored_docs.sort(key=lambda x: x[1], reverse=True)

        top_doc, top_score = scored_docs[0]

        # ---------------------------------------------------------
        # 8. Ambiguity band logic
        # ---------------------------------------------------------
        with start_trace("ambiguity_band_logic"):
            ambiguity_band = top_score - margin

            ambiguous_docs = [
                doc for (doc, score) in scored_docs
                if score >= ambiguity_band
            ]

            # Only one doc above band → auto-select
            if len(ambiguous_docs) == 1:
                return {
                    **state,
                    "selected_doc_id": top_doc["doc_id"],
                    "needs_user_choice": False,
                    "candidate_docs": [d for d, _ in scored_docs]
                }

            # Otherwise → HITL
            return {
                **state,
                "selected_doc_id": None,
                "needs_user_choice": True,
                "candidate_docs": [d for d, _ in scored_docs]
            }
