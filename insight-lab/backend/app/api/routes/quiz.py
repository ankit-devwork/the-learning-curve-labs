from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, Request

from pycorekit.tracing.decorators import with_observability

from app.core.auth import AuthUser
from app.core.deps import get_current_user
from app.core.supabase_client import get_supabase_client
from app.services.quiz_service import (
    generate_document_quiz,
    get_document_quiz,
    submit_quiz_attempt,
)

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
