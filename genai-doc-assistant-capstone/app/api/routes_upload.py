from fastapi import APIRouter, UploadFile, File, Request
from pathlib import Path
import uuid
import asyncio
import hashlib

from pycorekit.logging.logger import get_logger
from pycorekit.utils.uploader import upload_bytes
from pycorekit.exceptions.file import FileException
from pycorekit.exceptions.base import AppException

from pycorekit.tracing.decorators import with_observability
from pycorekit.tracing.tracing import start_trace
from pycorekit.correlation.context import get_current_correlation_id

from app.core.settings import settings
from app.core.parsers import parse_file_content
from app.core.llm import llm
from app.core.rag import (
    basic_clean,
    chunk_text,
    dedupe_semantic_async,
    filter_low_quality,
    dedupe_exact,
)
from app.service.db_connection import (
    async_embed_texts,
    async_add_to_chroma,
    async_get_collection,
)

router = APIRouter(tags=["Upload + Ingest"])
log = get_logger("upload_ingest")

# @with_observability wraps this endpoint and appends sanitized trace data.
# Endpoint handlers should not manually serialize or inject request.state.trace.

ALLOWED_EXTENSIONS = settings.file_upload.allowed_file_types
MAX_FILE_SIZE = settings.file_upload.max_file_size_mb * 1024 * 1024


def _flatten_metadatas(raw_metadatas):
    if not raw_metadatas:
        return []
    if isinstance(raw_metadatas, list) and raw_metadatas and isinstance(raw_metadatas[0], list):
        return [item for batch in raw_metadatas for item in batch if isinstance(item, dict)]
    return raw_metadatas


def validate_file(file: UploadFile, file_bytes: bytes):
    filename = file.filename

    if not filename:
        raise FileException("Uploaded file has no name")

    ext = filename.split(".")[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise FileException("Unsupported file type")

    if len(file_bytes) == 0:
        raise FileException("Uploaded file is empty")

    if len(file_bytes) > MAX_FILE_SIZE:
        raise FileException(
            message=f"File too large. Max allowed size is {MAX_FILE_SIZE} bytes"
        )


def _build_summary_prompt(title: str, chunks: list[str], cleaned: str, max_chunks: int = 8) -> str:
    """Build a compact summary prompt using a limited document excerpt."""
    if not chunks:
        summary_source = cleaned[:4000]
    else:
        excerpt = "\n\n".join(chunks[:max_chunks])
        summary_source = excerpt if excerpt else cleaned[:4000]

    return (
        "Summarize the following document in 3–5 sentences. "
        "Do not add any preamble such as 'Here is a summary...'.\n\n"
        f"Document title: {title}\n\n"
        f"Document content excerpt:\n{summary_source}"
    )


def _sanitize_summary(summary: str) -> str:
    """Normalize a summary by stripping prompt-like prefixes."""
    if not summary:
        return ""

    cleaned = summary.strip()

    prefixes = [
        "here is a summary of the document in 3-5 sentences:",
        "here is a summary of the document in 3–5 sentences:",
        "here is a summary of the document in 3-5 sentences",
        "here is a summary of the document in 3–5 sentences",
    ]

    lowered = cleaned.lower()
    for prefix in prefixes:
        if lowered.startswith(prefix):
            cleaned = cleaned[len(prefix):].lstrip(':').strip()
            break

    return cleaned


@router.post(
    "/upload-and-ingest",
    summary="Upload a document and ingest it into the vector database",
    description=(
        "Full ingestion pipeline:\n\n"
        "1. File validation\n"
        "2. File persistence\n"
        "3. Parsing\n"
        "4. Cleaning\n"
        "5. Chunking\n"
        "6. Quality filtering\n"
        "7. Deduplication\n"
        "8. Metadata generation\n"
        "9. Embedding\n"
        "10. Storage in ChromaDB\n"
    ),
    operation_id="uploadAndIngest",
)
@with_observability("upload_and_ingest")
async def upload_and_ingest(request: Request, file: UploadFile = File(...)):
    log.info(f"Received file: {file.filename}")

    # 1. Read + validate file
    with start_trace("validate_file", inputs={"filename": file.filename}):
        file_bytes = await file.read()
        validate_file(file, file_bytes)

    # 2. Duplicate detection (SHA-256)
    with start_trace("duplicate_detection", inputs={"filename": file.filename}):
        file_hash = hashlib.sha256(file_bytes).hexdigest()

        collection = await async_get_collection("documents")
        existing = await asyncio.to_thread(collection.get, include=["metadatas"])
        metadatas = _flatten_metadatas(existing.get("metadatas", []))

        for meta in metadatas:
            if meta.get("file_hash") == file_hash:
                return {
                    "message": "Duplicate file detected — skipping ingestion",
                    "duplicate": True,
                    "filename": file.filename,
                    "doc_id": meta["doc_id"],
                    "title": meta.get("title"),
                    "summary": meta.get("summary"),
                    "correlation_id": get_current_correlation_id() or "unknown",
                }

        safe_filename = Path(file.filename).name
        if not safe_filename:
            raise FileException("Invalid uploaded file name")

        saved_path = settings.paths.upload_dir / safe_filename
        saved_path_str = await upload_bytes(
            file_bytes,
            dest="local",
            dest_dir=str(settings.paths.upload_dir),
            dest_name=safe_filename,
        )
        saved_path = Path(saved_path_str)
        log.info(f"File saved to: {saved_path}")

    # 4. Parse file
    with start_trace("parse_file", inputs={"path": str(saved_path)}):
        try:
            content = await asyncio.to_thread(parse_file_content, saved_path)
        except Exception as e:
            raise AppException(
                message=f"Failed to parse file: {e}",
                status_code=400,
                error_type="INGEST_ERROR",
            )

    # 5. Clean text
    with start_trace("clean_text"):
        cleaned = basic_clean(str(content))

    # 6. Chunk text
    with start_trace("chunk_text"):
        chunks = chunk_text(
            cleaned,
            settings.rag.chunk_size,
            settings.rag.chunk_overlap,
            settings.rag.splitter,
        )

    # 7. Quality filtering
    with start_trace("filter_low_quality"):
        chunks = filter_low_quality(chunks, threshold=settings.rag.quality_threshold)

    # 8. Exact deduplication
    with start_trace("dedupe_exact"):
        chunks = dedupe_exact(chunks)

    # 9. Semantic deduplication (optional)
    if settings.rag.semantic_dedupe:
        with start_trace("dedupe_semantic_async"):
            chunks = await dedupe_semantic_async(chunks, async_embed_texts)

    # 10. Document metadata
    doc_id = str(uuid.uuid4())
    base_title = file.filename.rsplit(".", 1)[0]

    with start_trace("generate_summary"):
        summary_prompt = _build_summary_prompt(base_title, chunks, cleaned)
        summary = await llm(summary_prompt)
        summary = _sanitize_summary(summary)

    # 11. Embeddings
    with start_trace("embed_chunks_async"):
        embeddings = await async_embed_texts(chunks)

    # 12. Store in ChromaDB
    with start_trace("store_in_chroma_async"):
        metadata = [
            {
                "doc_id": doc_id,
                "title": base_title,
                "summary": summary,
                "filename": file.filename,
                "chunk_index": i,
                "path": str(saved_path),
                "file_hash": file_hash,
            }
            for i in range(len(chunks))
        ]

        await async_add_to_chroma(
            doc_id=doc_id,
            chunks=chunks,
            embeddings=embeddings,
            metadata=metadata,
        )

    result = {
        "message": "Upload + ingestion successful",
        "filename": file.filename,
        "saved_path": str(saved_path),
        "size_bytes": len(file_bytes),
        "doc_id": doc_id,
        "title": base_title,
        "summary": summary,
        "model": settings.models.llm_model,
        "num_chunks": len(chunks),
        "content_preview": cleaned[:200],
        "duplicate": False,
        "correlation_id": get_current_correlation_id() or "unknown",
    }

    log.info("Returning response", result=result)
    with start_trace("return_response", inputs={"correlation_id": get_current_correlation_id() or "unknown"}):
        return result
