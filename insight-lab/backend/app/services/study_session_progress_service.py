"""Persisted study session progress — start, resume, and complete steps."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pycorekit.exceptions.file import FileException
from supabase import Client

from app.core.auth import AuthUser
from app.core.exceptions import NotFoundException
from app.services.study_session_service import get_study_session_plan, get_workspace_study_session_plan
from app.services.workspace_access import require_workspace_role


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _serialize_session(
    session: dict[str, Any],
    steps: list[dict[str, Any]],
) -> dict[str, Any]:
    completed = sum(1 for step in steps if step.get("status") == "completed")
    total = len(steps)
    return {
        "session_id": session["id"],
        "session_type": session["session_type"],
        "workspace_id": session.get("workspace_id"),
        "document_id": session.get("document_id"),
        "learning_path_id": session.get("learning_path_id"),
        "status": session["status"],
        "current_step_index": session.get("current_step_index", 0),
        "started_at": session.get("started_at"),
        "last_activity_at": session.get("last_activity_at"),
        "completed_at": session.get("completed_at"),
        "progress": {
            "completed_steps": completed,
            "total_steps": total,
            "percent": round((completed / total) * 100, 1) if total else 0,
        },
        "plan": session.get("plan_snapshot") or {},
        "steps": steps,
    }


def _abandon_active_sessions(
    client: Client,
    *,
    user_id: str,
    workspace_id: str | None = None,
    document_id: str | None = None,
) -> None:
    query = (
        client.table("study_sessions")
        .select("id")
        .eq("user_id", user_id)
        .eq("status", "active")
    )
    if workspace_id:
        query = query.eq("workspace_id", workspace_id)
    if document_id:
        query = query.eq("document_id", document_id)
    rows = query.execute().data or []
    for row in rows:
        client.table("study_sessions").update(
            {"status": "abandoned", "last_activity_at": _now()}
        ).eq("id", row["id"]).execute()


def _persist_session(
    client: Client,
    *,
    user: AuthUser,
    session_type: str,
    plan: dict[str, Any],
    workspace_id: str | None = None,
    document_id: str | None = None,
    learning_path_id: str | None = None,
) -> dict[str, Any]:
    session_id = str(uuid4())
    steps = plan.get("steps") or []
    client.table("study_sessions").insert(
        {
            "id": session_id,
            "user_id": user.id,
            "workspace_id": workspace_id,
            "document_id": document_id,
            "learning_path_id": learning_path_id,
            "session_type": session_type,
            "status": "active",
            "plan_snapshot": plan,
            "current_step_index": 0,
            "started_at": _now(),
            "last_activity_at": _now(),
        }
    ).execute()

    step_rows = []
    for index, step in enumerate(steps):
        step_rows.append(
            {
                "id": str(uuid4()),
                "session_id": session_id,
                "step_index": index,
                "step_type": step.get("step", "unknown"),
                "label": step.get("label", f"Step {index + 1}"),
                "payload": step,
                "status": "pending",
            }
        )
    if step_rows:
        client.table("study_session_steps").insert(step_rows).execute()

    session = (
        client.table("study_sessions")
        .select("*")
        .eq("id", session_id)
        .limit(1)
        .execute()
        .data[0]
    )
    stored_steps = (
        client.table("study_session_steps")
        .select("id, step_index, step_type, label, payload, status, started_at, completed_at")
        .eq("session_id", session_id)
        .order("step_index")
        .execute()
        .data
        or []
    )
    return _serialize_session(session, stored_steps)


async def start_document_study_session(
    client: Client,
    document_id: str,
    user: AuthUser,
) -> dict[str, Any]:
    plan = await get_study_session_plan(client, document_id, user)
    doc = plan.get("document_id") or document_id
    _abandon_active_sessions(client, user_id=user.id, document_id=doc)
    return _persist_session(
        client,
        user=user,
        session_type="document",
        plan=plan,
        document_id=doc,
    )


async def start_workspace_study_session(
    client: Client,
    workspace_id: str,
    user: AuthUser,
    *,
    learning_path_id: str | None = None,
) -> dict[str, Any]:
    require_workspace_role(client, workspace_id, user, min_role="viewer")
    plan = await get_workspace_study_session_plan(client, workspace_id, user)
    if learning_path_id:
        plan["learning_path_id"] = learning_path_id
    _abandon_active_sessions(client, user_id=user.id, workspace_id=workspace_id)
    return _persist_session(
        client,
        user=user,
        session_type="workspace",
        plan=plan,
        workspace_id=workspace_id,
        learning_path_id=learning_path_id,
    )


def get_study_session(client: Client, session_id: str, user: AuthUser) -> dict[str, Any]:
    session = (
        client.table("study_sessions")
        .select("*")
        .eq("id", session_id)
        .eq("user_id", user.id)
        .limit(1)
        .execute()
        .data
    )
    if not session:
        raise NotFoundException("Study session not found")
    row = session[0]
    steps = (
        client.table("study_session_steps")
        .select("id, step_index, step_type, label, payload, status, started_at, completed_at")
        .eq("session_id", session_id)
        .order("step_index")
        .execute()
        .data
        or []
    )
    return _serialize_session(row, steps)


def get_active_workspace_study_session(
    client: Client,
    workspace_id: str,
    user: AuthUser,
) -> dict[str, Any] | None:
    require_workspace_role(client, workspace_id, user, min_role="viewer")
    session = (
        client.table("study_sessions")
        .select("*")
        .eq("user_id", user.id)
        .eq("workspace_id", workspace_id)
        .eq("status", "active")
        .order("last_activity_at", desc=True)
        .limit(1)
        .execute()
        .data
    )
    if not session:
        return None
    return get_study_session(client, session[0]["id"], user)


def advance_study_session_step(
    client: Client,
    session_id: str,
    user: AuthUser,
    *,
    step_index: int,
    status: str = "completed",
) -> dict[str, Any]:
    session = get_study_session(client, session_id, user)
    if session["status"] != "active":
        raise FileException("Study session is not active", status_code=409)

    step = (
        client.table("study_session_steps")
        .select("*")
        .eq("session_id", session_id)
        .eq("step_index", step_index)
        .limit(1)
        .execute()
        .data
    )
    if not step:
        raise NotFoundException("Study session step not found")

    now = _now()
    step_update: dict[str, Any] = {"status": status}
    if status == "in_progress" and not step[0].get("started_at"):
        step_update["started_at"] = now
    if status in ("completed", "skipped"):
        step_update["completed_at"] = now

    client.table("study_session_steps").update(step_update).eq("id", step[0]["id"]).execute()

    steps = (
        client.table("study_session_steps")
        .select("id, step_index, status")
        .eq("session_id", session_id)
        .order("step_index")
        .execute()
        .data
        or []
    )
    next_index = step_index
    if status == "completed":
        for row in steps:
            if row["status"] == "pending":
                next_index = row["step_index"]
                break
        else:
            next_index = step_index

    all_done = all(row["status"] in ("completed", "skipped") for row in steps)
    session_update: dict[str, Any] = {
        "current_step_index": next_index,
        "last_activity_at": now,
    }
    if all_done:
        session_update["status"] = "completed"
        session_update["completed_at"] = now
    client.table("study_sessions").update(session_update).eq("id", session_id).execute()

    return get_study_session(client, session_id, user)


def complete_study_session_step_for_quiz(
    client: Client,
    *,
    user: AuthUser,
    study_session_step_id: str | None,
    attempt_id: str | None,
) -> None:
    if not study_session_step_id:
        return
    step = (
        client.table("study_session_steps")
        .select("id, session_id, step_index")
        .eq("id", study_session_step_id)
        .limit(1)
        .execute()
        .data
    )
    if not step:
        return

    session = (
        client.table("study_sessions")
        .select("id, user_id")
        .eq("id", step[0]["session_id"])
        .limit(1)
        .execute()
        .data
    )
    if not session or session[0]["user_id"] != user.id:
        return

    if attempt_id:
        client.table("quiz_attempts").update(
            {"study_session_step_id": study_session_step_id}
        ).eq("id", attempt_id).execute()

    advance_study_session_step(
        client,
        step[0]["session_id"],
        user,
        step_index=step[0]["step_index"],
        status="completed",
    )
