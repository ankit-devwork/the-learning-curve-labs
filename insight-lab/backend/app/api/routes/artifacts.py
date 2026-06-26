from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, Request
from fastapi.responses import PlainTextResponse, Response

from pycorekit.tracing.decorators import with_observability
from pycorekit.correlation.headers import tracking_response_headers

from app.core.auth import AuthUser
from app.core.deps import get_current_user
from app.core.supabase_client import get_supabase_client
from app.services.artifact_service import (
    generate_document_flashcards,
    generate_document_infographic,
    generate_document_slide_deck,
    generate_document_study_guide,
    get_document_flashcards,
    get_document_infographic,
    get_document_slide_deck,
    get_document_study_guide,
    get_flashcard_set,
    get_slide_deck_by_id,
    get_study_guide_by_id,
    review_flashcard,
)
from app.services.audio_overview_service import (
    generate_audio_overview,
    get_audio_overview,
    get_audio_overview_mp3_bytes,
)
from app.services.explain_service import explain_flashcard, explain_quiz_question
from app.services.export_utils import flashcards_to_anki_csv, slide_deck_to_markdown, study_guide_to_markdown

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


@router.post("/documents/{document_id}/infographics/generate")
@with_observability("generate_infographic")
async def generate_infographic_route(
    document_id: str,
    request: Request,
    user: AuthUser = Depends(get_current_user),
):
    result = await generate_document_infographic(get_supabase_client(), document_id, user)
    correlation_id = getattr(request.state, "correlation_id", None)
    return {**result, "correlation_id": correlation_id}


@router.get("/documents/{document_id}/infographics")
@with_observability("get_infographic")
async def get_infographic_route(
    document_id: str,
    request: Request,
    user: AuthUser = Depends(get_current_user),
):
    result = await get_document_infographic(get_supabase_client(), document_id, user)
    correlation_id = getattr(request.state, "correlation_id", None)
    return {"infographic": result, "correlation_id": correlation_id}


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


@router.get("/documents/{document_id}/audio-overview/mp3")
@with_observability("get_audio_overview_mp3")
async def get_audio_overview_mp3_route(
    document_id: str,
    request: Request,
    user: AuthUser = Depends(get_current_user),
):
    mp3_bytes, filename = await get_audio_overview_mp3_bytes(
        get_supabase_client(),
        document_id,
        user,
    )
    correlation_id = getattr(request.state, "correlation_id", None)
    return Response(
        content=mp3_bytes,
        media_type="audio/mpeg",
        headers={
            "Content-Disposition": f'inline; filename="{filename}"',
            **tracking_response_headers(correlation_id),
        },
    )


@router.post("/documents/{document_id}/slide-decks/generate")
@with_observability("generate_slide_deck")
async def generate_slide_deck_route(
    document_id: str,
    request: Request,
    user: AuthUser = Depends(get_current_user),
):
    result = await generate_document_slide_deck(get_supabase_client(), document_id, user)
    correlation_id = getattr(request.state, "correlation_id", None)
    return {**result, "correlation_id": correlation_id}


@router.get("/documents/{document_id}/slide-decks")
@with_observability("get_slide_deck")
async def get_slide_deck_route(
    document_id: str,
    request: Request,
    user: AuthUser = Depends(get_current_user),
):
    result = await get_document_slide_deck(get_supabase_client(), document_id, user)
    correlation_id = getattr(request.state, "correlation_id", None)
    return {"slide_deck": result, "correlation_id": correlation_id}


@router.get("/slide-decks/{slide_deck_id}/export/markdown")
@with_observability("export_slide_deck_markdown")
async def export_slide_deck_markdown_route(
    slide_deck_id: str,
    request: Request,
    user: AuthUser = Depends(get_current_user),
):
    deck = await get_slide_deck_by_id(get_supabase_client(), slide_deck_id, user)
    markdown = slide_deck_to_markdown(title=deck["title"], content=deck["content"])
    correlation_id = getattr(request.state, "correlation_id", None)
    return PlainTextResponse(
        content=markdown,
        media_type="text/markdown; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="slides-{slide_deck_id}.md"',
            **tracking_response_headers(correlation_id),
        },
    )


@router.post("/quizzes/{quiz_id}/questions/{question_id}/explain")
@with_observability("explain_quiz_question")
async def explain_quiz_question_route(
    quiz_id: str,
    question_id: str,
    request: Request,
    user: AuthUser = Depends(get_current_user),
):
    result = await explain_quiz_question(
        get_supabase_client(),
        quiz_id,
        question_id,
        user,
    )
    correlation_id = getattr(request.state, "correlation_id", None)
    return {**result, "correlation_id": correlation_id}


@router.post("/flashcards/{set_id}/cards/{flashcard_id}/explain")
@with_observability("explain_flashcard")
async def explain_flashcard_route(
    set_id: str,
    flashcard_id: str,
    request: Request,
    user: AuthUser = Depends(get_current_user),
):
    result = await explain_flashcard(
        get_supabase_client(),
        set_id,
        flashcard_id,
        user,
    )
    correlation_id = getattr(request.state, "correlation_id", None)
    return {**result, "correlation_id": correlation_id}


@router.get("/flashcards/{set_id}/export/anki")
@with_observability("export_flashcards_anki")
async def export_flashcards_anki_route(
    set_id: str,
    request: Request,
    user: AuthUser = Depends(get_current_user),
):
    flashcards = await get_flashcard_set(get_supabase_client(), set_id, user)
    csv_text = flashcards_to_anki_csv(flashcards.get("cards") or [])
    correlation_id = getattr(request.state, "correlation_id", None)
    return PlainTextResponse(
        content=csv_text,
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="flashcards-{set_id}.csv"',
            **tracking_response_headers(correlation_id),
        },
    )


@router.get("/study-guides/{guide_id}/export/markdown")
@with_observability("export_study_guide_markdown")
async def export_study_guide_markdown_route(
    guide_id: str,
    request: Request,
    user: AuthUser = Depends(get_current_user),
):
    guide = await get_study_guide_by_id(get_supabase_client(), guide_id, user)
    markdown = study_guide_to_markdown(title=guide["title"], content=guide["content"])
    correlation_id = getattr(request.state, "correlation_id", None)
    return PlainTextResponse(
        content=markdown,
        media_type="text/markdown; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="study-guide-{guide_id}.md"',
            **tracking_response_headers(correlation_id),
        },
    )
