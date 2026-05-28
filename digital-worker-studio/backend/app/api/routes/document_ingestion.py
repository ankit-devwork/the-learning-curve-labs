import os
import uuid
import shutil
import json
from typing import List

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from pypdf import PdfReader
from litellm import acompletion
from sentence_transformers import SentenceTransformer

# Project Aligned Services and Core Architecture Frameworks
from app.services.graph_synchronizer import sync_document_to_knowledge_graph
from app.services.cache_service import cache_service  # 🚀 NEW: Integrated Cache Engine Singleton
from app.core.database import db_service
from app.models.document import DocumentModel, DocumentStatus
from app.models.document_chunk import DocumentChunkModel
from app.schemas.document import DocumentInsightsSchema
from app.core.load_property import settings
from app.observability.logger import logger

# OCR stack handling (optional dependency fallback checking)
try:
    from pdf2image import convert_from_path
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

router = APIRouter(prefix="/api/worker/document", tags=["Document Ingestion"])

# Configuration Constants mapped from properties infrastructure
TEXT_EMBEDDING_MODEL = settings.text_embedding_model  # e.g., "sentence-transformers/all-mpnet-base-v2"
BASE_GENERATION_MODEL = settings.base_model            # e.g., "groq/llama-3.3-70b-versatile"
CHUNK_SIZE = settings.chunk_size
CHUNK_OVERLAP = settings.chunk_overlap

# File size boundary settings mapped from core workspace settings (converted to bytes)
MAX_FILE_SIZE_MB = settings.storage_max_file_size_mb
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

logger.info(f"Loading embedding model into system memory: {TEXT_EMBEDDING_MODEL}")
local_vector_engine = SentenceTransformer(TEXT_EMBEDDING_MODEL)
local_vector_engine.max_seq_length = 512


# -----------------------------------------------------------------------------------------
# TEXT CHUNKING UTILITY
# -----------------------------------------------------------------------------------------
def chunk_text_sliding_window(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Splits continuous document text matrices into uniform chunks using a sliding window strategy.
    
    Args:
        text (str): Raw string representation extracted from target source layout.
        chunk_size (int): Max text character characters per block constraint boundary.
        overlap (int): Number of overlapping characters to reserve at sliding window intersection points.
        
    Returns:
        List[str]: Cleaned list elements of structured text segment fragments.
    """
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
    """Parses standard high-level text character layers out of natively compiled PDF assets."""
    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        txt = page.extract_text() or ""
        if txt.strip():
            pages.append(txt)
    return "\n\n".join(pages)


def extract_text_ocr(path: str) -> str:
    """Fallback OCR process leveraging Tesseract to pull text matrices from rasterized/scanned PDFs."""
    if not OCR_AVAILABLE:
        raise RuntimeError("OCR requested but pdf2image or pytesseract libraries are not installed in this environment.")
    images = convert_from_path(path)
    texts = []
    for img in images:
        txt = pytesseract.image_to_string(img) or ""
        if txt.strip():
            texts.append(txt)
    return "\n\n".join(texts)


# -----------------------------------------------------------------------------------------
# EMBEDDING + DB PERSISTENCE (pgvector Vector Track)
# -----------------------------------------------------------------------------------------
async def embed_chunks_and_persist(db: AsyncSession, document_id: str, chunks: List[str]) -> int:
    """
    Batch encodes text chunks using the local embedding model and commits entries to PostgreSQL.
    
    Args:
        db (AsyncSession): Active SQLAlchemy async transaction worker thread connection handle.
        document_id (str): The primary key string routing back to the core file ledger tracking record.
        chunks (List[str]): Extracted sliding window text chunks to process.
    """
    if not chunks:
        return 0

    # Execute native matrix encoding leveraging local Hugging Face weights inside system memory
    embeddings = local_vector_engine.encode(
        chunks,
        batch_size=16,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    for idx, vector in enumerate(embeddings):
        db.add(
            DocumentChunkModel(
                document_id=document_id,
                chunk_index=idx,
                content=chunks[idx],
                embedding=vector.tolist(),  # Cast numpy array back to native list float representations
            )
        )
    
    await db.flush()
    return len(chunks)


# -----------------------------------------------------------------------------------------
# GROQ INTEL EXTRACTION ENGINE (Graph Track)
# -----------------------------------------------------------------------------------------
async def run_groq_extraction(full_text: str) -> dict:
    """
    Submits document text segments to the Groq Cloud endpoint for schema extraction via LiteLLM.
    
    Incorporates inference instructions that force the model to look at structural headers
    and paragraphs contextually to dynamically extract assignees and relative timelines.
    """
    trimmed_context = full_text[:30000] 

    system_instruction = (
        "You are an elite, context-aware corporate intelligence extraction engine.\n"
        "Analyze the provided document text segment and extract an executive summary, "
        "a list of critical risks, and key operational tasks matching the schema parameters.\n\n"
        "CRITICAL RULES FOR INTERPRETING METADATA CONSTRAINTS:\n"
        "1. **Inferred Assignees:** If a task is listed under a specific heading or section "
        "(e.g., 'MetLife Legal Plan Provisions' or 'Employer Requirements'), do not leave the assignee "
        "as 'Unassigned'. Deduce the logical owner from the structural context of the paragraph "
        "(e.g., 'MetLife', 'The Employee', or 'HR Administrator').\n"
        "2. **Dynamic / Relative Deadlines:** Real documents rarely use hard calendar dates. "
        "Look for and aggressively extract relative timelines and temporal constraints! Examples to capture:\n"
        "   - 'During the open enrollment window'\n"
        "   - 'Within 30 days of contract execution'\n"
        "   - 'Prior to effective coverage activation'\n"
        "   - 'On-going / As needed'\n"
        "3. **Zero Default Fallbacks:** Only resort to using 'None Specified' or 'Unassigned' if there is "
        "absolutely zero textual, situational, or contextual clue within the document text segment."
    )

    response = await acompletion(
        model=BASE_GENERATION_MODEL,
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": f"Document text segment content:\n\n{trimmed_context}"},
        ],
        api_key=os.getenv("GROQ_API_KEY"),
        response_format=DocumentInsightsSchema,
    )

    raw_content = response.choices[0].message.content
    if isinstance(raw_content, str):
        return json.loads(raw_content)
    return raw_content


# -----------------------------------------------------------------------------------------
# CORE INGESTION ROUTE HANDLER
# -----------------------------------------------------------------------------------------
@router.post("/upload")
async def initial_file_upload(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(db_service.get_session),
):
    """
    Primary multi-track entry-point for document ingestion.
    
    Saves multi-part files onto disk, generates overlapping chunks for pgvector searches, 
    extracts structured entities via Groq, maps knowledge topology elements into Neo4j, 
    and automatically flushes stale query cache blocks out of Redis storage memory.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Unsupported media extension file format. Only PDF configurations are supported.")

    # Guardrail: Instant file size validation checkpoint check at the HTTP server gateway
    if file.size > MAX_FILE_SIZE_BYTES:
        current_size_mb = file.size / (1024 * 1024)
        logger.warning(f"Rejected storage payload allocation limit overflow: {file.filename} ({current_size_mb:.2f} MB)")
        raise HTTPException(
            status_code=413,
            detail=f"File payload too large. Maximum size allowed is {MAX_FILE_SIZE_MB} MB. Uploaded: {current_size_mb:.2f} MB."
        )

    upload_dir = settings.storage_local_dir
    os.makedirs(upload_dir, exist_ok=True)

    document_id = str(uuid.uuid4())
    safe_filename = f"{document_id}_{file.filename}"
    local_file_path = os.path.join(upload_dir, safe_filename)

    logger.info(f"[{document_id}] Starting unified data track ingestion pipeline pass...")

    try:
        # Stream multi-part payload content bits onto stable local storage volumes
        with open(local_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Create relational tracker row log profile inside PostgreSQL mapping state
        db_doc = DocumentModel(
            id=document_id,
            filename=file.filename,
            file_path=local_file_path,
            status=DocumentStatus.PROCESSING,
        )
        db.add(db_doc)
        await db.commit()
        
        # Safely re-fetch active relational state profiles to secure worker persistence threads
        db_doc = await db.get(DocumentModel, document_id)

        # 1. TEXT EXTRACTION LAYER RUN
        logger.info(f"[{document_id}] Extracting text layers via PyPDF...")
        full_text = extract_text_pypdf(local_file_path)

        if not full_text.strip():
            logger.warning(f"[{document_id}] Emptied plain-text layer encountered. Directing execution to OCR fallback...")
            full_text = extract_text_ocr(local_file_path)

        if not full_text.strip():
            raise ValueError("PDF document does not contain readable characters, vector printing, or image elements.")

        # 2. VECTOR STORAGE PIPELINE RUN (PostgreSQL pgvector)
        chunks = chunk_text_sliding_window(full_text, CHUNK_SIZE, CHUNK_OVERLAP)
        logger.info(f"[{document_id}] Sliced target text matrix into {len(chunks)} blocks.")
        total_chunks = await embed_chunks_and_persist(db, document_id, chunks)

        # 3. KNOWLEDGE TOPOLOGY PIPELINE RUN (Neo4j Graph Database)
        logger.info(f"[{document_id}] Launching Groq structured inference tracking block...")
        try:
            extracted = await run_groq_extraction(full_text)
            db_doc.extracted_data = extracted
            
            # Sync extraction insights down into structural Neo4j Graph layouts
            await sync_document_to_knowledge_graph(
                document_id=document_id,
                filename=file.filename,
                insights=extracted
            )
        except Exception as err:
            logger.error(f"[{document_id}] Structural parsing layer failed: {err}")
            db_doc.extracted_data = {
                "executive_summary": "Document insight extraction crashed during the Groq parsing phase.",
                "risks": [],
                "tasks_and_deadlines": [],
            }

        # 4. DATA SYNCHRONIZATION AND STATUS FINALIZATION
        db_doc.status = DocumentStatus.COMPLETED
        await db.commit()

        # 5. 🚀 REDIS CACHE INVALIDATION GATEWAY EXECUTOR
        # Clears stale query response vectors to ensure real-time analysis parity
        await cache_service.flush_all_query_caches()

        return {
            "status": "success",
            "document_id": document_id,
            "total_chunks_vectorized": total_chunks,
            "extraction_status": "synced",
            "message": "Pipeline processing completed successfully. Graph sync completed and stale query cache flushed.",
        }

    except Exception as e:
        logger.error(f"[{document_id}] Ingestion pipeline critically collapsed: {e}")
        await db.rollback()

        try:
            # Re-fetch instance logging row elements to safely track trace failure parameters
            db_doc = await db.get(DocumentModel, document_id)
            if db_doc:
                db_doc.status = DocumentStatus.FAILED
                db_doc.error_message = str(e)
                await db.commit()
        except Exception as rollback_err:
            logger.error(f"Failed to commit crash state log parameters to target database metadata: {rollback_err}")

        raise HTTPException(500, f"Ingestion pipeline execution fault breakdown: {str(e)}")