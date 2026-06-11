"""
retriever.py
------------
Async + observable Retriever Agent.

Responsibilities:
- Use selected_doc_id (if present) to scope retrieval to a single document.
- Query Chroma for top‑k chunks relevant to the user question.
"""

import asyncio
from app.service.db_connection import (
    async_embed_texts,
    async_get_collection
)
from app.core.agent_state import AgentState

from pycorekit.logging.logger import get_logger

# Unified tracing engine
from pycorekit.tracing.tracing import start_trace

# DB observability wrapper
from pycorekit.observability.observability import observe_db

from pycorekit.correlation.context import get_current_correlation_id

log = get_logger("retriever")


async def retriever_agent(state: AgentState):
    cid = get_current_correlation_id() or "unknown"

    with start_trace("retriever_agent", inputs={"state": dict(state)}):
        log.info("Retriever agent started")

        question = state["question"]
        selected_doc_id = state.get("selected_doc_id")

        # 1. Embed question
        with start_trace("embed_question", inputs={"question": question}):
            query_embedding = (await async_embed_texts([question]))[0]

        # 2. Get collection
        with start_trace("get_collection_async"):
            collection = await async_get_collection("documents")

        # 3. Build query args
        query_args = {
            "query_embeddings": [query_embedding],
            "n_results": 5,
        }
        if selected_doc_id:
            query_args["where"] = {"doc_id": selected_doc_id}

        # 4. Query Chroma (DB observability)
        results = await observe_db(
            "chroma_query_async",
            func=lambda: asyncio.to_thread(collection.query, **query_args),
            query=str(query_args)
        )

        # 5. Extract chunks
        docs = results.get("documents", [[]])
        metas = results.get("metadatas", [[]])

        retrieved_docs = docs[0] if docs else []
        retrieved_metas = metas[0] if metas else []

        # 6. Prepend stored summary if doc_id is selected
        if selected_doc_id:
            with start_trace("fetch_summary_metadata"):
                summary_results = await observe_db(
                    "chroma_get_summary",
                    func=lambda: asyncio.to_thread(
                        collection.get,
                        where={"doc_id": selected_doc_id},
                        include=["metadatas"]
                    ),
                    query=f"collection.get(where={{'doc_id': '{selected_doc_id}'}})"
                )

                if summary_results and summary_results.get("metadatas"):
                    summary_meta = summary_results["metadatas"][0]
                    if "summary" in summary_meta:
                        retrieved_docs.insert(0, summary_meta["summary"])
                        retrieved_metas.insert(0, summary_meta)

        log.info(f"Retriever agent finished. Retrieved {len(retrieved_docs)} chunks")

        return {
            **state,
            "retrieved_chunks": retrieved_docs,
            "retrieved_metadata": retrieved_metas,
        }
