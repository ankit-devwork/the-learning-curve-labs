from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, Query, Request

from pycorekit.tracing.decorators import with_observability

from app.core.auth import AuthUser
from app.core.deps import get_current_user
from app.core.supabase_client import get_supabase_client
from app.services.artifact_service import (
    generate_document_flashcards,
    generate_document_study_guide,
    get_document_flashcards,
    get_document_study_guide,
    review_flashcard,
)
from app.services.audio_overview_service import generate_audio_overview, get_audio_overview

router = APIRouter(tags=["artifacts"])


class FlashcardGenerateRequest(BaseModel):
    num_cards: int = Field(default=10, ge=3, le=20)


class FlashcardReviewRequest(BaseModel):
    flashcard_id: str = Field(..., min_length=1)
    knew: bool


@router.post("/documents/{document_id}/flashcards/generate")
@with_observability("generate_flashcards")
async def generate_flashcards_route(
    document_id: str,
    body: FlashcardGenerateRequest,
    request: Request,
    user: AuthUser = Depends(get_current_user),
):
    result = await generate_document_flashcards(
        get_supabase_client(),
        document_id,
        user,
        num_cards=body.num_cards,
    )
    correlation_id = getattr(request.state, "correlation_id", None)
    return {**result, "correlation_id": correlation_id}


@router.get("/documents/{document_id}/flashcards")
@with_observability("get_flashcards")
async def get_flashcards_route(
    document_id: str,
    request: Request,
    user: AuthUser = Depends(get_current_user),
):
    result = await get_document_flashcards(get_supabase_client(), document_id, user)
    correlation_id = getattr(request.state, "correlation_id", None)
    return {"flashcards": result, "correlation_id": correlation_id}


@router.post("/flashcards/{set_id}/review")
@with_observability("review_flashcard")
async def review_flashcard_route(
    set_id: str,
    body: FlashcardReviewRequest,
    request: Request,
    user: AuthUser = Depends(get_current_user),
):
    result = await review_flashcard(
        get_supabase_client(),
        set_id,
        user,
        flashcard_id=body.flashcard_id,
        knew=body.knew,
    )
    correlation_id = getattr(request.state, "correlation_id", None)
    return {**result, "correlation_id": correlation_id}


@router.post("/documents/{document_id}/study-guide/generate")
@with_observability("generate_study_guide")
async def generate_study_guide_route(
    document_id: str,
    request: Request,
    user: AuthUser = Depends(get_current_user),
):
    result = await generate_document_study_guide(get_supabase_client(), document_id, user)
    correlation_id = getattr(request.state, "correlation_id", None)
    return {**result, "correlation_id": correlation_id}


@router.get("/documents/{document_id}/study-guide")
@with_observability("get_study_guide")
async def get_study_guide_route(
    document_id: str,
    request: Request,
    user: AuthUser = Depends(get_current_user),
):
    result = await get_document_study_guide(get_supabase_client(), document_id, user)
    correlation_id = getattr(request.state, "correlation_id", None)
    return {"study_guide": result, "correlation_id": correlation_id}


@router.post("/documents/{document_id}/audio-overview/generate")
@with_observability("generate_audio_overview")
async def generate_audio_overview_route(
    document_id: str,
    request: Request,
    user: AuthUser = Depends(get_current_user),
):
    result = await generate_audio_overview(get_supabase_client(), document_id, user)
    correlation_id = getattr(request.state, "correlation_id", None)
    return {**result, "correlation_id": correlation_id}


@router.get("/documents/{document_id}/audio-overview")
@with_observability("get_audio_overview")
async def get_audio_overview_route(
    document_id: str,
    request: Request,
    user: AuthUser = Depends(get_current_user),
):
    result = await get_audio_overview(get_supabase_client(), document_id, user)
    correlation_id = getattr(request.state, "correlation_id", None)
    if result is None:
        return {
            "document_id": document_id,
            "audio_overview": None,
            "correlation_id": correlation_id,
        }
    return {"audio_overview": result, "correlation_id": correlation_id}
