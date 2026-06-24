from pydantic import BaseModel, Field
from fastapi import APIRouter, Request

from pycorekit.tracing.decorators import with_observability

from app.core.supabase_client import get_supabase_client
from app.services.quiz_service import get_public_quiz, submit_public_quiz

router = APIRouter(prefix="/public/quizzes", tags=["public-quiz"])


class PublicSubmitQuizRequest(BaseModel):
    display_name: str = Field(default="Guest", max_length=80)
    answers: dict[str, int] = Field(default_factory=dict)


@router.get("/{share_token}")
@with_observability("get_public_quiz")
async def get_public_quiz_route(share_token: str, request: Request):
    result = await get_public_quiz(get_supabase_client(), share_token=share_token)
    correlation_id = getattr(request.state, "correlation_id", None)
    return {**result, "correlation_id": correlation_id}


@router.post("/{share_token}/submit")
@with_observability("submit_public_quiz")
async def submit_public_quiz_route(
    share_token: str,
    body: PublicSubmitQuizRequest,
    request: Request,
):
    result = await submit_public_quiz(
        get_supabase_client(),
        share_token=share_token,
        display_name=body.display_name,
        answers=body.answers,
    )
    correlation_id = getattr(request.state, "correlation_id", None)
    return {**result, "correlation_id": correlation_id}
