from fastapi import APIRouter, Request
import asyncio

from pycorekit.core_logging.logger import get_logger
from pycorekit.tracing.decorators import with_observability
from pycorekit.tracing.tracing import start_trace
from pycorekit.correlation.context import get_current_correlation_id
from pycorekit.observability.observability import observe_db


def _flatten_metadatas(raw_metadatas):
    if not raw_metadatas:
        return []
    if isinstance(raw_metadatas, list) and raw_metadatas and isinstance(raw_metadatas[0], list):
        return [item for batch in raw_metadatas for item in batch if isinstance(item, dict)]
    return raw_metadatas


from app.service.db_connection import get_collection

router = APIRouter(tags=["Query"])
log = get_logger("list_documents")

# The decorated route will append a sanitized observability trace
# to JSON dict responses after the handler completes.


@router.get(
    "/documents",
    summary="List all uploaded documents",
    operation_id="listDocuments"
)
@with_observability("list_documents")
async def list_documents(request: Request):

    cid = get_current_correlation_id() or "unknown"

    # 1. Get collection (NOT async, NOT a DB call)
    with start_trace("fetch_documents_collection"):
        collection = get_collection("documents")

        # Wrap ONLY the DB call
        docs = await observe_db(
            "chroma_get_documents",
            func=lambda: asyncio.to_thread(
                collection.get,
                include=["metadatas"]
            ),
            query="collection.get(include=['metadatas'])"
        )

    # 2. Handle empty
    with start_trace("check_empty"):
        raw_metadatas = docs.get("metadatas", [])
        metadatas = _flatten_metadatas(raw_metadatas)
        raw_metadata_shape = (
            "list-of-list"
            if isinstance(raw_metadatas, list) and raw_metadatas and isinstance(raw_metadatas[0], list)
            else type(raw_metadatas).__name__
        )
        raw_metadata_count = (
            sum(len(batch) for batch in raw_metadatas)
            if isinstance(raw_metadatas, list) and raw_metadatas and isinstance(raw_metadatas[0], list)
            else len(raw_metadatas) if isinstance(raw_metadatas, list)
            else 0
        )

        if not docs or not metadatas:
            return {
                "status": "ok",
                "documents": [],
                "count": 0,
                "raw_metadata_shape": raw_metadata_shape,
                "raw_metadata_count": raw_metadata_count,
                "correlation_id": cid,
            }

    # 3. Deduplicate
    with start_trace("dedupe_documents"):
        doc_map = {}
        for meta in metadatas:
            doc_id = meta["doc_id"]
            if doc_id not in doc_map:
                doc_map[doc_id] = {
                    "doc_id": doc_id,
                    "filename": meta.get("filename", "unknown"),
                    "title": meta.get("title", "Untitled Document"),
                    "summary": meta.get("summary", "")
                }

        documents = list(doc_map.values())

    # 4. Return
    with start_trace("format_response"):
        return {
            "status": "ok",
            "documents": documents,
            "count": len(documents),
            "raw_metadata_shape": raw_metadata_shape,
            "raw_metadata_count": raw_metadata_count,
            "correlation_id": cid,
        }
