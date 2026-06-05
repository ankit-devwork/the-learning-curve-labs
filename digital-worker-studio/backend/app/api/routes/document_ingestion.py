import os
import uuid
import shutil
import json
from typing import List

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession
from pypdf import PdfReader
import aiofiles
import asyncio
from litellm import acompletion
from sentence_transformers import SentenceTransformer

# Project Aligned Services and Core Architecture Frameworks
from app.services.graph_synchronizer import sync_document_to_knowledge_graph
from app.services.cache_service import cache_service
from app.core.database import db_service
from app.models.document import DocumentModel, DocumentStatus
from app.models.document_chunk import DocumentChunkModel
from app.schemas.document import DocumentInsightsSchema
from app.core.load_property import settings
from app.observability.logger import get_request_logger

# OCR stack handling (optional dependency fallback checking)
try:
    from pdf2image import convert_from_path
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

router = APIRouter(prefix="/api/worker/document", tags=["Document Ingestion"])

TEXT_EMBEDDING_MODEL = settings.text_embedding_model
BASE_GENERATION_MODEL = settings.base_model
CHUNK_SIZE = settings.chunk_size
CHUNK_OVERLAP = settings.chunk_overlap

MAX_FILE_SIZE_MB = settings.storage_max_file_size_mb
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

local_vector_engine = SentenceTransformer(TEXT_EMBEDDING_MODEL)
try:
    local_vector_engine.max_seq_length = 512
except Exception:
    pass


# -----------------------------------------------------------------------------------------
# TEXT CHUNKING UTILITY
# -----------------------------------------------------------------------------------------
def chunk_text_sliding_window(text: str, chunk_size: int, overlap: int) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip().replace("\n", " ")
        if chunk:
            chunks.append(chunk)
        start += (chunk_size - overlap)
    return chunks


# -----------------------------------------------------------------------------------------
# PDF TEXT EXTRACTION PIPELINES
# -----------------------------------------------------------------------------------------
def extract_text_pypdf(path: str) -> str:
    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        txt = page.extract_text() or ""
        if txt.strip():
            pages.append(txt)
    return "\n\n".join(pages)


def extract_text_ocr(path: str) -> str:
    if not OCR_AVAILABLE:
        raise RuntimeError("OCR requested but pdf2image or pytesseract not installed.")
    images = convert_from_path(path)
    texts = []
    for img in images:
        txt = pytesseract.image_to_string(img) or ""
        if txt.strip():
            texts.append(txt)
    return "\n\n".join(texts)


# -----------------------------------------------------------------------------------------
# EMBEDDING + DB PERSISTENCE
# -----------------------------------------------------------------------------------------
async def embed_chunks_and_persist(db: AsyncSession, document_id: str, chunks: List[str]) -> int:
    if not chunks:
        return 0

    def _encode():
        return local_vector_engine.encode(
            chunks,
            batch_size=16,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    embeddings = await asyncio.to_thread(_encode)

    models = []
    for idx, vector in enumerate(embeddings):
        models.append(
            DocumentChunkModel(
                document_id=document_id,
                chunk_index=idx,
                content=chunks[idx],
                embedding=vector.tolist(),
            )
        )

    db.add_all(models)
    await db.flush()
    return len(models)


# -----------------------------------------------------------------------------------------
# GROQ EXTRACTION ENGINE
# -----------------------------------------------------------------------------------------
async def run_groq_extraction(full_text: str) -> dict:
    trimmed_context = full_text[:30000]

    system_instruction = (
        "You are an elite, context-aware corporate intelligence extraction engine.\n"
        "Analyze the provided document text segment and extract an executive summary, "
        "a list of critical risks, and key operational tasks matching the schema parameters.\n"
    )

    response = await acompletion(
        model=BASE_GENERATION_MODEL,
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": f"Document text:\n\n{trimmed_context}"},
        ],
        api_key=os.getenv("GROQ_API_KEY"),
        response_format=DocumentInsightsSchema,
    )

    raw = response.choices[0].message.content
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            return {
                "executive_summary": "Extraction returned invalid JSON.",
                "risks": [],
                "tasks_and_deadlines": [],
            }
    return raw


# -----------------------------------------------------------------------------------------
# CORE INGESTION ROUTE HANDLER (PATCHED)
# -----------------------------------------------------------------------------------------
@router.post("/upload")
async def initial_file_upload(
    request: Request,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(db_service.get_session),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")

    upload_dir = settings.storage_local_dir
    os.makedirs(upload_dir, exist_ok=True)

    document_id = str(uuid.uuid4())
    safe_filename = f"{document_id}_{file.filename}"
    local_file_path = os.path.join(upload_dir, safe_filename)

    # Bind logger with correlation_id + document_id
    log = get_request_logger(request, document_id=document_id)

    total_bytes = 0
    chunk_size = 1024 * 64

    try:
        async with aiofiles.open(local_file_path, "wb") as out_f:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > MAX_FILE_SIZE_BYTES:
                    raise HTTPException(
                        413,
                        f"File too large. Max {MAX_FILE_SIZE_MB} MB allowed.",
                    )
                await out_f.write(chunk)

    except Exception as write_err:
        log.error(f"[{document_id}] Failed to save uploaded file: {write_err}")
        raise HTTPException(500, "Failed to save uploaded file")

    log.info(f"[{document_id}] Starting unified ingestion pipeline...")

    try:
        db_doc = DocumentModel(
            id=document_id,
            filename=file.filename,
            file_path=local_file_path,
            status=DocumentStatus.PROCESSING,
        )
        db.add(db_doc)
        await db.commit()
        await db.refresh(db_doc)

        log.info(f"[{document_id}] Extracting text via PyPDF...")
        full_text = await asyncio.to_thread(extract_text_pypdf, local_file_path)

        if not full_text.strip():
            log.warning(f"[{document_id}] Empty text layer. Trying OCR fallback...")
            full_text = await asyncio.to_thread(extract_text_ocr, local_file_path)

        if not full_text.strip():
            raise HTTPException(400, "PDF contains no readable text.")

        chunks = chunk_text_sliding_window(full_text, CHUNK_SIZE, CHUNK_OVERLAP)
        log.info(f"[{document_id}] Chunked into {len(chunks)} blocks.")

        total_chunks = await embed_chunks_and_persist(db, document_id, chunks)

        log.info(f"[{document_id}] Running Groq extraction...")
        try:
            extracted = await run_groq_extraction(full_text)
            db_doc.extracted_data = extracted

            await sync_document_to_knowledge_graph(
                document_id=document_id,
                filename=file.filename,
                insights=extracted,
            )
        except Exception as err:
            log.error(f"[{document_id}] Groq extraction failed: {err}")
            db_doc.extracted_data = {
                "executive_summary": "Extraction failed.",
                "risks": [],
                "tasks_and_deadlines": [],
            }

        db_doc.status = DocumentStatus.COMPLETED
        await db.commit()

        try:
            await cache_service.flush_all_query_caches()
        except Exception as cache_err:
            log.warning(f"[{document_id}] Cache flush failed: {cache_err}")

        return {
            "status": "success",
            "document_id": document_id,
            "total_chunks_vectorized": total_chunks,
            "extraction_status": "synced",
            "message": "Ingestion completed successfully.",
        }

    except Exception as e:
        log.error(f"[{document_id}] Pipeline crashed: {e}")
        await db.rollback()

        try:
            db_doc = await db.get(DocumentModel, document_id)
            if db_doc:
                db_doc.status = DocumentStatus.FAILED
                db_doc.error_message = str(e)
                await db.commit()
        except Exception:
            pass

        raise HTTPException(500, f"Ingestion failed: {str(e)}")
