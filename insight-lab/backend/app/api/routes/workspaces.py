from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, Query, Request

from pycorekit.tracing.decorators import with_observability

from app.api.routes.quiz import GenerateQuizRequest
from app.core.auth import AuthUser
from app.core.deps import get_current_user
from app.core.supabase_client import get_supabase_client
from app.services.mastery_service import get_workspace_concept_mastery, get_workspace_weak_concepts
from app.services.quiz_service import generate_workspace_adaptive_quiz
from app.services.workspace_service import (
    create_workspace,
    delete_workspace,
    get_workspace,
    get_workspace_stats,
    list_workspace_documents,
    list_workspaces,
    update_workspace,
)

router = APIRouter(prefix="/workspaces", tags=["workspaces"])


class WorkspaceCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: str | None = Field(default=None, max_length=500)


class WorkspaceUpdateRequest(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=100)
    description: str | None = Field(default=None, max_length=500)


@router.get("")
@with_observability("list_workspaces")
async def list_workspaces_route(
    request: Request,
    user: AuthUser = Depends(get_current_user),
):
    workspaces = list_workspaces(get_supabase_client(), user)
    correlation_id = getattr(request.state, "correlation_id", None)
    return {"workspaces": workspaces, "count": len(workspaces), "correlation_id": correlation_id}


@router.post("")
@with_observability("create_workspace")
async def create_workspace_route(
    body: WorkspaceCreateRequest,
    request: Request,
    user: AuthUser = Depends(get_current_user),
):
    workspace = create_workspace(
        get_supabase_client(),
        user,
        name=body.name,
        description=body.description,
    )
    correlation_id = getattr(request.state, "correlation_id", None)
    return {**workspace, "correlation_id": correlation_id}


@router.get("/{workspace_id}")
@with_observability("get_workspace")
async def get_workspace_route(
    workspace_id: str,
    request: Request,
    user: AuthUser = Depends(get_current_user),
):
    workspace = get_workspace(get_supabase_client(), workspace_id, user)
    correlation_id = getattr(request.state, "correlation_id", None)
    return {**workspace, "correlation_id": correlation_id}


@router.patch("/{workspace_id}")
@with_observability("update_workspace")
async def update_workspace_route(
    workspace_id: str,
    body: WorkspaceUpdateRequest,
    request: Request,
    user: AuthUser = Depends(get_current_user),
):
    workspace = update_workspace(
        get_supabase_client(),
        workspace_id,
        user,
        name=body.name,
        description=body.description,
    )
    correlation_id = getattr(request.state, "correlation_id", None)
    return {**workspace, "correlation_id": correlation_id}


@router.delete("/{workspace_id}")
@with_observability("delete_workspace")
async def delete_workspace_route(
    workspace_id: str,
    request: Request,
    user: AuthUser = Depends(get_current_user),
):
    delete_workspace(get_supabase_client(), workspace_id, user)
    correlation_id = getattr(request.state, "correlation_id", None)
    return {"deleted": True, "workspace_id": workspace_id, "correlation_id": correlation_id}


@router.get("/{workspace_id}/documents")
@with_observability("list_workspace_documents")
async def list_workspace_documents_route(
    workspace_id: str,
    request: Request,
    user: AuthUser = Depends(get_current_user),
    limit: int = Query(default=50, ge=1, le=100),
):
    documents = list_workspace_documents(
        get_supabase_client(),
        workspace_id,
        user,
        limit=limit,
    )
    correlation_id = getattr(request.state, "correlation_id", None)
    return {"documents": documents, "count": len(documents), "correlation_id": correlation_id}


@router.get("/{workspace_id}/stats")
@with_observability("workspace_stats")
async def workspace_stats_route(
    workspace_id: str,
    request: Request,
    user: AuthUser = Depends(get_current_user),
):
    stats = get_workspace_stats(get_supabase_client(), workspace_id, user)
    correlation_id = getattr(request.state, "correlation_id", None)
    return {**stats, "correlation_id": correlation_id}


@router.get("/{workspace_id}/concepts/mastery")
@with_observability("get_workspace_concept_mastery")
async def get_workspace_concept_mastery_route(
    workspace_id: str,
    request: Request,
    user: AuthUser = Depends(get_current_user),
):
    result = await get_workspace_concept_mastery(get_supabase_client(), workspace_id, user)
    correlation_id = getattr(request.state, "correlation_id", None)
    return {**result, "correlation_id": correlation_id}


@router.get("/{workspace_id}/concepts/weak")
@with_observability("get_workspace_weak_concepts")
async def get_workspace_weak_concepts_route(
    workspace_id: str,
    request: Request,
    user: AuthUser = Depends(get_current_user),
):
    concepts = await get_workspace_weak_concepts(get_supabase_client(), workspace_id, user)
    correlation_id = getattr(request.state, "correlation_id", None)
    return {
        "workspace_id": workspace_id,
        "concepts": concepts,
        "correlation_id": correlation_id,
    }


@router.post("/{workspace_id}/quiz/adaptive/generate")
@with_observability("generate_workspace_adaptive_quiz")
async def generate_workspace_adaptive_quiz_route(
    workspace_id: str,
    body: GenerateQuizRequest,
    request: Request,
    user: AuthUser = Depends(get_current_user),
):
    result = await generate_workspace_adaptive_quiz(
        get_supabase_client(),
        workspace_id,
        user,
        question_type=body.question_type,
        difficulty=body.difficulty,
        num_questions=body.num_questions,
    )
    correlation_id = getattr(request.state, "correlation_id", None)
    return {**result, "correlation_id": correlation_id}
