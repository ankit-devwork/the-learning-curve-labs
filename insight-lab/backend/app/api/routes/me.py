from fastapi import APIRouter, Depends, Request

from pycorekit.tracing.decorators import with_observability

from app.core.auth import AuthUser
from app.core.deps import get_current_user

router = APIRouter()


@router.get("/me")
@with_observability("get_current_user_profile")
async def get_me(
    request: Request,
    user: AuthUser = Depends(get_current_user),
):
    correlation_id = getattr(request.state, "correlation_id", None)
    return {
        "user_id": user.id,
        "email": user.email,
        "role": user.role,
        "authenticated": True,
        "correlation_id": correlation_id,
    }
