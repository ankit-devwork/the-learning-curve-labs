from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, Request
from fastapi.responses import PlainTextResponse

from pycorekit.tracing.decorators import with_observability
from pycorekit.correlation.headers import tracking_response_headers

from app.core.auth import AuthUser
from app.core.deps import get_current_user
from app.core.supabase_client import get_supabase_client
from app.services.export_utils import quiz_to_qti_xml
from app.services.quiz_service import (
    generate_adaptive_quiz,
    generate_document_quiz,
    get_document_quiz,
    publish_quiz,
    submit_quiz_attempt,
    update_quiz_question,
    get_quiz_for_edit,
)
from app.services.mastery_service import get_concept_mastery, get_weak_concepts

router = APIRouter()


class GenerateQuizRequest(BaseModel):
    question_type: str = Field(default="scq", pattern=r"^(scq|mcq|true_false)$")
    difficulty: str = Field(default="medium", pattern=r"^(easy|medium|hard)$")
    num_questions: int = Field(default=5, ge=1, le=20)


class SubmitQuizRequest(BaseModel):
    answers: dict[str, int] = Field(default_factory=dict)


class UpdateQuizQuestionRequest(BaseModel):
    question_text: str | None = Field(default=None, min_length=1, max_length=2000)
    options: list[str] | None = Field(default=None, min_length=2, max_length=6)
    correct_option_index: int | None = Field(default=None, ge=0)
    explanation: str | None = Field(default=None, max_length=4000)


@router.post("/documents/{document_id}/quiz/generate")
@with_observability("generate_document_quiz")
async def generate_document_quiz_route(
    document_id: str,
    body: GenerateQuizRequest,
    request: Request,
    user: AuthUser = Depends(get_current_user),
):
    result = await generate_document_quiz(
        get_supabase_client(),
        document_id,
        user,
        question_type=body.question_type,
        difficulty=body.difficulty,
        num_questions=body.num_questions,
    )
    correlation_id = getattr(request.state, "correlation_id", None)
    return {**result, "correlation_id": correlation_id}


@router.get("/documents/{document_id}/quiz")
@with_observability("get_document_quiz")
async def get_document_quiz_route(
    document_id: str,
    request: Request,
    user: AuthUser = Depends(get_current_user),
):
    result = await get_document_quiz(get_supabase_client(), document_id, user)
    correlation_id = getattr(request.state, "correlation_id", None)
    if result is None:
        return {
            "document_id": document_id,
            "quiz": None,
            "correlation_id": correlation_id,
        }
    return {**result, "correlation_id": correlation_id}


@router.post("/quizzes/{quiz_id}/submit")
@with_observability("submit_quiz_attempt")
async def submit_quiz_attempt_route(
    quiz_id: str,
    body: SubmitQuizRequest,
    request: Request,
    user: AuthUser = Depends(get_current_user),
):
    result = await submit_quiz_attempt(
        get_supabase_client(),
        quiz_id,
        user,
        answers=body.answers,
    )
    correlation_id = getattr(request.state, "correlation_id", None)
    return {**result, "correlation_id": correlation_id}


@router.post("/documents/{document_id}/quiz/adaptive/generate")
@with_observability("generate_adaptive_quiz")
async def generate_adaptive_quiz_route(
    document_id: str,
    body: GenerateQuizRequest,
    request: Request,
    user: AuthUser = Depends(get_current_user),
):
    result = await generate_adaptive_quiz(
        get_supabase_client(),
        document_id,
        user,
        question_type=body.question_type,
        difficulty=body.difficulty,
        num_questions=body.num_questions,
    )
    correlation_id = getattr(request.state, "correlation_id", None)
    return {**result, "correlation_id": correlation_id}


@router.get("/documents/{document_id}/concepts/mastery")
@with_observability("get_concept_mastery")
async def get_concept_mastery_route(
    document_id: str,
    request: Request,
    user: AuthUser = Depends(get_current_user),
):
    result = await get_concept_mastery(get_supabase_client(), document_id, user)
    correlation_id = getattr(request.state, "correlation_id", None)
    return {**result, "correlation_id": correlation_id}


@router.get("/documents/{document_id}/concepts/weak")
@with_observability("get_weak_concepts")
async def get_weak_concepts_route(
    document_id: str,
    request: Request,
    user: AuthUser = Depends(get_current_user),
):
    concepts = await get_weak_concepts(get_supabase_client(), document_id, user)
    correlation_id = getattr(request.state, "correlation_id", None)
    return {
        "document_id": document_id,
        "concepts": concepts,
        "correlation_id": correlation_id,
    }


@router.get("/quizzes/{quiz_id}/edit")
@with_observability("get_quiz_for_edit")
async def get_quiz_for_edit_route(
    quiz_id: str,
    request: Request,
    user: AuthUser = Depends(get_current_user),
):
    result = await get_quiz_for_edit(get_supabase_client(), quiz_id, user)
    correlation_id = getattr(request.state, "correlation_id", None)
    return {**result, "correlation_id": correlation_id}


@router.patch("/quizzes/{quiz_id}/questions/{question_id}")
@with_observability("update_quiz_question")
async def update_quiz_question_route(
    quiz_id: str,
    question_id: str,
    body: UpdateQuizQuestionRequest,
    request: Request,
    user: AuthUser = Depends(get_current_user),
):
    result = await update_quiz_question(
        get_supabase_client(),
        quiz_id,
        question_id,
        user,
        question_text=body.question_text,
        options=body.options,
        correct_option_index=body.correct_option_index,
        explanation=body.explanation,
    )
    correlation_id = getattr(request.state, "correlation_id", None)
    return {**result, "correlation_id": correlation_id}


@router.post("/quizzes/{quiz_id}/publish")
@with_observability("publish_quiz")
async def publish_quiz_route(
    quiz_id: str,
    request: Request,
    user: AuthUser = Depends(get_current_user),
):
    result = await publish_quiz(get_supabase_client(), quiz_id, user)
    correlation_id = getattr(request.state, "correlation_id", None)
    return {**result, "correlation_id": correlation_id}


@router.get("/quizzes/{quiz_id}/export/qti")
@with_observability("export_quiz_qti")
async def export_quiz_qti_route(
    quiz_id: str,
    request: Request,
    user: AuthUser = Depends(get_current_user),
):
    quiz = await get_quiz_for_edit(get_supabase_client(), quiz_id, user)
    xml_text = quiz_to_qti_xml(title=quiz["title"], questions=quiz["questions"])
    correlation_id = getattr(request.state, "correlation_id", None)
    return PlainTextResponse(
        content=xml_text,
        media_type="application/xml; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="quiz-{quiz_id}.xml"',
            **tracking_response_headers(correlation_id),
        },
    )
