from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, Query, Request

from pycorekit.tracing.decorators import with_observability

from app.api.routes.quiz import GenerateQuizRequest
from app.core.auth import AuthUser
from app.core.deps import get_current_user
from app.core.supabase_client import get_supabase_client
from app.services.course_pack_service import generate_course_pack
from app.services.mastery_service import get_workspace_concept_mastery, get_workspace_weak_concepts
from app.services.quiz_service import generate_workspace_adaptive_quiz
from app.services.sharing_service import (
    create_workspace_invite,
    leave_workspace,
    list_workspace_invites,
    list_workspace_members,
    remove_workspace_member,
    revoke_workspace_invite,
    update_member_role,
)
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


class WorkspaceInviteRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=320)
    role: str = Field(default="viewer", pattern=r"^(editor|viewer)$")


class MemberRoleUpdateRequest(BaseModel):
    role: str = Field(..., pattern=r"^(editor|viewer)$")


class CoursePackRequest(BaseModel):
    document_ids: list[str] | None = Field(default=None, max_length=20)


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


@router.post("/{workspace_id}/course-pack/generate")
@with_observability("generate_course_pack")
async def generate_course_pack_route(
    workspace_id: str,
    body: CoursePackRequest,
    request: Request,
    user: AuthUser = Depends(get_current_user),
):
    result = await generate_course_pack(
        get_supabase_client(),
        workspace_id,
        user,
        document_ids=body.document_ids,
    )
    correlation_id = getattr(request.state, "correlation_id", None)
    return {**result, "correlation_id": correlation_id}


@router.get("/{workspace_id}/members")
@with_observability("list_workspace_members")
async def list_workspace_members_route(
    workspace_id: str,
    request: Request,
    user: AuthUser = Depends(get_current_user),
):
    members = list_workspace_members(get_supabase_client(), workspace_id, user)
    correlation_id = getattr(request.state, "correlation_id", None)
    return {"members": members, "correlation_id": correlation_id}


@router.get("/{workspace_id}/invites")
@with_observability("list_workspace_invites")
async def list_workspace_invites_route(
    workspace_id: str,
    request: Request,
    user: AuthUser = Depends(get_current_user),
):
    invites = list_workspace_invites(get_supabase_client(), workspace_id, user)
    correlation_id = getattr(request.state, "correlation_id", None)
    return {"invites": invites, "correlation_id": correlation_id}


@router.post("/{workspace_id}/invites")
@with_observability("create_workspace_invite")
async def create_workspace_invite_route(
    workspace_id: str,
    body: WorkspaceInviteRequest,
    request: Request,
    user: AuthUser = Depends(get_current_user),
):
    invite = await create_workspace_invite(
        get_supabase_client(),
        workspace_id,
        user,
        email=body.email,
        role=body.role,
    )
    correlation_id = getattr(request.state, "correlation_id", None)
    return {**invite, "correlation_id": correlation_id}


@router.delete("/{workspace_id}/members/{member_user_id}")
@with_observability("remove_workspace_member")
async def remove_workspace_member_route(
    workspace_id: str,
    member_user_id: str,
    request: Request,
    user: AuthUser = Depends(get_current_user),
):
    await remove_workspace_member(
        get_supabase_client(),
        workspace_id,
        user,
        member_user_id=member_user_id,
    )
    correlation_id = getattr(request.state, "correlation_id", None)
    return {"deleted": True, "correlation_id": correlation_id}


@router.delete("/{workspace_id}/invites/{invite_id}")
@with_observability("revoke_workspace_invite")
async def revoke_workspace_invite_route(
    workspace_id: str,
    invite_id: str,
    request: Request,
    user: AuthUser = Depends(get_current_user),
):
    await revoke_workspace_invite(
        get_supabase_client(),
        workspace_id,
        user,
        invite_id=invite_id,
    )
    correlation_id = getattr(request.state, "correlation_id", None)
    return {"deleted": True, "correlation_id": correlation_id}


@router.post("/{workspace_id}/leave")
@with_observability("leave_workspace")
async def leave_workspace_route(
    workspace_id: str,
    request: Request,
    user: AuthUser = Depends(get_current_user),
):
    await leave_workspace(get_supabase_client(), workspace_id, user)
    correlation_id = getattr(request.state, "correlation_id", None)
    return {"left": True, "workspace_id": workspace_id, "correlation_id": correlation_id}


@router.patch("/{workspace_id}/members/{member_user_id}")
@with_observability("update_member_role")
async def update_member_role_route(
    workspace_id: str,
    member_user_id: str,
    body: MemberRoleUpdateRequest,
    request: Request,
    user: AuthUser = Depends(get_current_user),
):
    member = await update_member_role(
        get_supabase_client(),
        workspace_id,
        user,
        member_user_id=member_user_id,
        role=body.role,
    )
    correlation_id = getattr(request.state, "correlation_id", None)
    return {**member, "correlation_id": correlation_id}
