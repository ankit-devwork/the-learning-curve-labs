from fastapi import APIRouter, Depends, Request

from pycorekit.tracing.decorators import with_observability

from app.core.auth import AuthUser
from app.core.deps import get_current_user
from app.core.supabase_client import get_supabase_client
from app.services.excel_service import analyze_excel, get_excel_analysis

router = APIRouter()


@router.post("/documents/{document_id}/analyze")
@with_observability("analyze_excel")
async def analyze_excel_route(
    document_id: str,
    request: Request,
    user: AuthUser = Depends(get_current_user),
):
    result = await analyze_excel(get_supabase_client(), document_id, user)
    correlation_id = getattr(request.state, "correlation_id", None)
    return {**result, "correlation_id": correlation_id}


@router.get("/documents/{document_id}/charts")
@with_observability("get_excel_charts")
async def get_excel_charts_route(
    document_id: str,
    request: Request,
    user: AuthUser = Depends(get_current_user),
):
    result = await get_excel_analysis(get_supabase_client(), document_id, user)
    correlation_id = getattr(request.state, "correlation_id", None)
    return {**result, "correlation_id": correlation_id}
