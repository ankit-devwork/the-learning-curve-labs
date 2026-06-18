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
from app.services.multi_doc_service import ask_multiple_documents, retrieve_multiple_documents

router = APIRouter()


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


class ApprovedSourceRef(BaseModel):
    document_id: str
    chunk_index: int = Field(ge=0)
    similarity: float | None = None


class MultiRetrieveRequest(BaseModel):
    document_ids: list[str] = Field(..., min_length=1, max_length=10)
    question: str = Field(..., min_length=1, max_length=2000)


class MultiAskRequest(BaseModel):
    document_ids: list[str] = Field(..., min_length=1, max_length=10)
    question: str = Field(..., min_length=1, max_length=2000)
    approved_sources: list[ApprovedSourceRef] = Field(..., min_length=1, max_length=20)


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
        approved_sources=[ref.model_dump() for ref in body.approved_sources],
    )
    correlation_id = getattr(request.state, "correlation_id", None)
    return {**result, "correlation_id": correlation_id}


@router.get("/documents/{document_id}")