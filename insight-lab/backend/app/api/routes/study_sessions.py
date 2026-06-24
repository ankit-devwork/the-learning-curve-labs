from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, Request

from pycorekit.tracing.decorators import with_observability
from pycorekit.correlation.headers import tracking_response_headers

from app.api.routes.quiz import GenerateQuizRequest
from app.core.auth import AuthUser
from app.core.deps import get_current_user
from app.core.supabase_client import get_supabase_client
from app.services.study_session_progress_service import (
    advance_study_session_step,
    get_study_session,
)

router = APIRouter(prefix="/study-sessions", tags=["study-sessions"])


class CompleteStepRequest(BaseModel):
    status: str = Field(default="completed", pattern=r"^(in_progress|completed|skipped)$")


@router.get("/{session_id}")
@with_observability("get_study_session")
async def get_study_session_route(
    session_id: str,
    request: Request,
    user: AuthUser = Depends(get_current_user),
):
    result = get_study_session(get_supabase_client(), session_id, user)
    correlation_id = getattr(request.state, "correlation_id", None)
    return {**result, "correlation_id": correlation_id}


@router.post("/{session_id}/steps/{step_index}/advance")
@with_observability("advance_study_session_step")
async def advance_study_session_step_route(
    session_id: str,
    step_index: int,
    body: CompleteStepRequest,
    request: Request,
    user: AuthUser = Depends(get_current_user),
):
    result = advance_study_session_step(
        get_supabase_client(),
        session_id,
        user,
        step_index=step_index,
        status=body.status,
    )
    correlation_id = getattr(request.state, "correlation_id", None)
    return {**result, "correlation_id": correlation_id}
