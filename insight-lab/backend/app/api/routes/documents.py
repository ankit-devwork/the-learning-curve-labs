from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, Request

from pycorekit.tracing.decorators import with_observability

from app.core.auth import AuthUser
from app.core.deps import get_current_user
from app.core.supabase_client import get_supabase_client
from app.services.document_service import (
    ask_document,
    get_document,
    get_document_summary,
    process_document,
)
from app.services.document_extras import (
    get_document_chunk,
    get_document_processing_status,
    get_suggested_questions,
)
from app.services.multi_doc_service import ask_multiple_documents, retrieve_multiple_documents

router = APIRouter()


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


class MultiRetrieveRequest(BaseModel):
    document_ids: list[str] = Field(..., min_length=1, max_length=10)
    question: str = Field(..., min_length=1, max_length=2000)


class MultiAskRequest(BaseModel):
    document_ids: list[str] = Field(..., min_length=1, max_length=10)
    question: str = Field(..., min_length=1, max_length=2000)
    approved_document_ids: list[str] = Field(..., min_length=1, max_length=10)


@router.post("/documents/multi/retrieve")
@with_observability("retrieve_multiple_documents")
async def retrieve_multiple_documents_route(
    body: MultiRetrieveRequest,
    request: Request,
    user: AuthUser = Depends(get_current_user),
):
    result = await retrieve_multiple_documents(
        get_supabase_client(),
        user,
        document_ids=body.document_ids,
        question=body.question,
    )
    correlation_id = getattr(request.state, "correlation_id", None)
    return {**result, "correlation_id": correlation_id}


@router.post("/documents/multi/ask")
@with_observability("ask_multiple_documents")
async def ask_multiple_documents_route(
    body: MultiAskRequest,
    request: Request,
    user: AuthUser = Depends(get_current_user),
):
    result = await ask_multiple_documents(
        get_supabase_client(),
        user,
        document_ids=body.document_ids,
        question=body.question,
        approved_document_ids=body.approved_document_ids,
    )
    correlation_id = getattr(request.state, "correlation_id", None)
    return {**result, "correlation_id": correlation_id}


@router.get("/documents/{document_id}")
@with_observability("get_document")
async def read_document(
    document_id: str,
    request: Request,
    user: AuthUser = Depends(get_current_user),
):
    document = await get_document(get_supabase_client(), document_id, user)
    correlation_id = getattr(request.state, "correlation_id", None)
    return {**document, "correlation_id": correlation_id}


@router.post("/documents/{document_id}/process")
@with_observability("process_document")
async def process_document_route(
    document_id: str,
    request: Request,
    user: AuthUser = Depends(get_current_user),
):
    result = await process_document(get_supabase_client(), document_id, user)
    correlation_id = getattr(request.state, "correlation_id", None)
    return {**result, "correlation_id": correlation_id}


@router.get("/documents/{document_id}/summary")
@with_observability("get_document_summary")
async def read_document_summary(
    document_id: str,
    request: Request,
    user: AuthUser = Depends(get_current_user),
):
    summary = await get_document_summary(get_supabase_client(), document_id, user)
    correlation_id = getattr(request.state, "correlation_id", None)
    return {**summary, "correlation_id": correlation_id}


@router.post("/documents/{document_id}/ask")
@with_observability("ask_document")
async def ask_document_route(
    document_id: str,
    body: AskRequest,
    request: Request,
    user: AuthUser = Depends(get_current_user),
):
    answer = await ask_document(
        get_supabase_client(),
        document_id,
        user,
        question=body.question,
    )
    correlation_id = getattr(request.state, "correlation_id", None)
    return {**answer, "correlation_id": correlation_id}


@router.get("/documents/{document_id}/status")
@with_observability("document_processing_status")
async def document_processing_status_route(
    document_id: str,
    request: Request,
    user: AuthUser = Depends(get_current_user),
):
    status = await get_document_processing_status(get_supabase_client(), document_id, user)
    correlation_id = getattr(request.state, "correlation_id", None)
    return {**status, "correlation_id": correlation_id}


@router.get("/documents/{document_id}/suggested-questions")
@with_observability("document_suggested_questions")
async def document_suggested_questions_route(
    document_id: str,
    request: Request,
    user: AuthUser = Depends(get_current_user),
):
    result = await get_suggested_questions(get_supabase_client(), document_id, user)
    correlation_id = getattr(request.state, "correlation_id", None)
    return {**result, "correlation_id": correlation_id}


@router.get("/documents/{document_id}/chunks/{chunk_index}")
@with_observability("get_document_chunk")
async def get_document_chunk_route(
    document_id: str,
    chunk_index: int,
    request: Request,
    user: AuthUser = Depends(get_current_user),
):
    chunk = await get_document_chunk(get_supabase_client(), document_id, chunk_index, user)
    correlation_id = getattr(request.state, "correlation_id", None)
    return {**chunk, "correlation_id": correlation_id}
