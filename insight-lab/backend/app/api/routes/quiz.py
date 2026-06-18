from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, Request

from pycorekit.tracing.decorators import with_observability

from app.core.auth import AuthUser
from app.core.deps import get_current_user
from app.core.supabase_client import get_supabase_client
from app.services.quiz_service import (
    generate_adaptive_quiz,
    generate_document_quiz,
    get_document_quiz,
    submit_quiz_attempt,
)
from app.services.mastery_service import get_concept_mastery, get_weak_concepts

router = APIRouter()


class GenerateQuizRequest(BaseModel):
    question_type: str = Field(default="scq", pattern=r"^(scq|mcq|true_false)$")
    difficulty: str = Field(default="medium", pattern=r"^(easy|medium|hard)$")
    num_questions: int = Field(default=5, ge=1, le=20)


class SubmitQuizRequest(BaseModel):
    answers: dict[str, int] = Field(default_factory=dict)


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
