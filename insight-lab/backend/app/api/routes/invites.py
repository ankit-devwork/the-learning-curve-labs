from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, Request

from pycorekit.tracing.decorators import with_observability

from app.core.auth import AuthUser
from app.core.deps import get_current_user
from app.core.supabase_client import get_supabase_client
from app.services.sharing_service import accept_workspace_invite, get_invite_preview

router = APIRouter(prefix="/invites", tags=["invites"])


@router.get("/{token}")
@with_observability("get_invite_preview")
async def get_invite_preview_route(token: str, request: Request):
    result = get_invite_preview(get_supabase_client(), token)
    correlation_id = getattr(request.state, "correlation_id", None)
    return {**result, "correlation_id": correlation_id}


@router.post("/{token}/accept")
@with_observability("accept_workspace_invite")
async def accept_workspace_invite_route(
    token: str,
    request: Request,
    user: AuthUser = Depends(get_current_user),
):
    result = accept_workspace_invite(get_supabase_client(), user, token=token)
    correlation_id = getattr(request.state, "correlation_id", None)
    return {**result, "correlation_id": correlation_id}
