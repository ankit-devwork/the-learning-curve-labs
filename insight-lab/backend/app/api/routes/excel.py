from fastapi import APIRouter, Depends, Request

from pycorekit.tracing.decorators import with_observability

from app.api.routes.documents import AskRequest
from app.core.auth import AuthUser
from app.core.deps import get_current_user
from app.core.supabase_client import get_supabase_client
from app.services.excel_charts import CustomChartRequest
from app.services.excel_service import (
    analyze_excel,
    ask_excel,
    create_custom_excel_chart,
    get_excel_analysis,
)

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


@router.post("/documents/{document_id}/charts/custom")
@with_observability("create_custom_excel_chart")
async def create_custom_excel_chart_route(
    document_id: str,
    body: CustomChartRequest,
    request: Request,
    user: AuthUser = Depends(get_current_user),
):
    result = await create_custom_excel_chart(get_supabase_client(), document_id, user, body)
    correlation_id = getattr(request.state, "correlation_id", None)
    return {**result, "correlation_id": correlation_id}


@router.post("/documents/{document_id}/excel/ask")
@with_observability("ask_excel")
async def ask_excel_route(
    document_id: str,
    body: AskRequest,
    request: Request,
    user: AuthUser = Depends(get_current_user),
):
    result = await ask_excel(
        get_supabase_client(),
        document_id,
        user,
        question=body.question,
    )
    correlation_id = getattr(request.state, "correlation_id", None)
    return {**result, "correlation_id": correlation_id}
