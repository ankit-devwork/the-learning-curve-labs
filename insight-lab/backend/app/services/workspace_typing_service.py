"""Member-scoped team chat typing presence (RLS-backed, backend heartbeats)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from supabase import Client

from app.core.auth import AuthUser
from app.core.cache import check_rate_limit
from app.core.exceptions import RateLimitException
from app.core.migration_guard import (
    PHASE3_023_MIGRATION_NOTICE,
    is_missing_team_chat_typing_schema,
)
from app.core.yaml_config import get_yaml_config
from app.services.workspace_access import require_workspace_role


def _author_label(profile: dict | None) -> str:
    if not profile:
        return "Member"
    display = (profile.get("display_name") or "").strip()
    if display:
        return display
    email = (profile.get("email") or "").strip()
    if email and "@" in email:
        return email.split("@", 1)[0]
    return "Member"


async def _check_typing_rate(user_id: str, workspace_id: str) -> None:
    cfg = get_yaml_config().team_chat
    allowed, retry_after = await check_rate_limit(
        key=f"team_chat:typing:{user_id}:{workspace_id}",
        limit=cfg.typing_rate_limit_per_min,
        window_seconds=60,
    )
    if not allowed:
        raise RateLimitException(
            "Too many typing updates; try again shortly",
            retry_after=retry_after,
        )


async def set_workspace_typing(
    client: Client,
    workspace_id: str,
    user: AuthUser,
    *,
    active: bool,
) -> dict:
    require_workspace_role(client, workspace_id, user, min_role="viewer")
    await _check_typing_rate(user.id, workspace_id)

    def _write() -> dict:
        if not active:
            client.table("workspace_typing_presence").delete().eq(
                "workspace_id", workspace_id
            ).eq("user_id", user.id).execute()
            return {"active": False}

        profile = (
            client.table("profiles")
            .select("id, display_name, email")
            .eq("id", user.id)
            .limit(1)
            .execute()
            .data
            or [None]
        )[0]
        now = datetime.now(timezone.utc).isoformat()
        client.table("workspace_typing_presence").upsert(
            {
                "workspace_id": workspace_id,
                "user_id": user.id,
                "display_name": _author_label(profile),
                "updated_at": now,
            },
            on_conflict="workspace_id,user_id",
        ).execute()
        return {"active": True}

    try:
        return _write()
    except Exception as exc:
        if is_missing_team_chat_typing_schema(exc):
            return {"active": False, "migration_required": True, "notice": PHASE3_023_MIGRATION_NOTICE}
        raise


async def list_workspace_typing(client: Client, workspace_id: str, user: AuthUser) -> dict:
    require_workspace_role(client, workspace_id, user, min_role="viewer")

    cfg = get_yaml_config().team_chat
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=cfg.typing_ttl_seconds)

    def _load() -> dict:
        rows = (
            client.table("workspace_typing_presence")
            .select("user_id, display_name, updated_at")
            .eq("workspace_id", workspace_id)
            .gt("updated_at", cutoff.isoformat())
            .neq("user_id", user.id)
            .execute()
            .data
            or []
        )
        typers = [
            {
                "user_id": row["user_id"],
                "name": row["display_name"] or "Member",
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]
        return {"typers": typers}

    try:
        return _load()
    except Exception as exc:
        if is_missing_team_chat_typing_schema(exc):
            return {"typers": [], "migration_required": True, "notice": PHASE3_023_MIGRATION_NOTICE}
        raise
