"""Study set invites and member management."""

from __future__ import annotations

from datetime import datetime, timezone

from pycorekit.exceptions.file import FileException
from supabase import Client

from app.core.auth import AuthUser
from app.core.cache import check_rate_limit
from app.core.exceptions import NotFoundException, RateLimitException
from app.core.yaml_config import get_yaml_config
from app.services.email_service import send_invite_email
from app.services.cache_invalidation import invalidate_user_workspace_access_caches
from app.services.upload import ensure_profile
from app.services.workspace_access import get_workspace_membership_role, require_workspace_role


async def _check_member_change_rate(user_id: str) -> None:
    cfg = get_yaml_config().sharing
    allowed, retry_after = await check_rate_limit(
        key=f"sharing:member_change:{user_id}",
        limit=cfg.member_change_rate_limit_per_min,
        window_seconds=60,
    )
    if not allowed:
        raise RateLimitException(
            f"Too many membership changes ({cfg.member_change_rate_limit_per_min}/min)",
            retry_after=retry_after,
        )


async def _check_invite_create_rate(user_id: str) -> None:
    cfg = get_yaml_config().sharing
    allowed, retry_after = await check_rate_limit(
        key=f"sharing:invite_create:{user_id}",
        limit=cfg.invite_create_rate_limit_per_min,
        window_seconds=60,
    )
    if not allowed:
        raise RateLimitException(
            f"Invite limit reached ({cfg.invite_create_rate_limit_per_min}/min)",
            retry_after=retry_after,
        )


async def _check_invite_preview_rate(token: str) -> None:
    cfg = get_yaml_config().sharing
    token_key = token.strip()[:16] or "unknown"
    allowed, retry_after = await check_rate_limit(
        key=f"sharing:invite_preview:{token_key}",
        limit=cfg.invite_preview_rate_limit_per_min,
        window_seconds=60,
    )
    if not allowed:
        raise RateLimitException(
            f"Too many invite preview requests ({cfg.invite_preview_rate_limit_per_min}/min)",
            retry_after=retry_after,
        )


async def _check_invite_accept_rate(user_id: str) -> None:
    cfg = get_yaml_config().sharing
    allowed, retry_after = await check_rate_limit(
        key=f"sharing:invite_accept:{user_id}",
        limit=cfg.invite_accept_rate_limit_per_min,
        window_seconds=60,
    )
    if not allowed:
        raise RateLimitException(
            f"Invite accept limit reached ({cfg.invite_accept_rate_limit_per_min}/min)",
            retry_after=retry_after,
        )


def list_workspace_members(client: Client, workspace_id: str, user: AuthUser) -> list[dict]:
    require_workspace_role(client, workspace_id, user, min_role="viewer")
    rows = (
        client.table("workspace_members")
        .select("workspace_id, user_id, role, joined_at, invited_by")
        .eq("workspace_id", workspace_id)
        .order("joined_at")
        .execute()
        .data
        or []
    )
    if not rows:
        return []

    user_ids = [row["user_id"] for row in rows]
    profiles = (
        client.table("profiles")
        .select("id, email, display_name")
        .in_("id", user_ids)
        .execute()
        .data
        or []
    )
    profile_map = {row["id"]: row for row in profiles}
    return [
        {
            **row,
            "email": profile_map.get(row["user_id"], {}).get("email"),
            "full_name": profile_map.get(row["user_id"], {}).get("display_name"),
        }
        for row in rows
    ]


def list_workspace_invites(client: Client, workspace_id: str, user: AuthUser) -> list[dict]:
    require_workspace_role(client, workspace_id, user, min_role="editor")
    return (
        client.table("workspace_invites")
        .select("id, email, role, token, expires_at, accepted_at, created_at")
        .eq("workspace_id", workspace_id)
        .is_("accepted_at", "null")
        .order("created_at", desc=True)
        .execute()
        .data
        or []
    )


async def create_workspace_invite(
    client: Client,
    workspace_id: str,
    user: AuthUser,
    *,
    email: str,
    role: str = "viewer",
) -> dict:
    await _check_invite_create_rate(user.id)
    ensure_profile(client, user)
    require_workspace_role(client, workspace_id, user, min_role="editor")
    normalized_email = email.strip().lower()
    if not normalized_email or "@" not in normalized_email:
        raise FileException("Valid email is required")
    if role not in {"editor", "viewer"}:
        raise FileException("Invite role must be editor or viewer")

    pending = (
        client.table("workspace_invites")
        .select("id")
        .eq("workspace_id", workspace_id)
        .eq("email", normalized_email)
        .is_("accepted_at", "null")
        .limit(1)
        .execute()
    )
    if pending.data:
        raise FileException("An invite is already pending for this email", status_code=409)

    existing_member = (
        client.table("profiles")
        .select("id")
        .eq("email", normalized_email)
        .limit(1)
        .execute()
    )
    if existing_member.data:
        member_id = existing_member.data[0]["id"]
        already = (
            client.table("workspace_members")
            .select("role")
            .eq("workspace_id", workspace_id)
            .eq("user_id", member_id)
            .limit(1)
            .execute()
        )
        if already.data:
            raise FileException("This user is already a member of the study set", status_code=409)

    row = {
        "workspace_id": workspace_id,
        "email": normalized_email,
        "role": role,
        "invited_by": user.id,
    }
    try:
        inserted = client.table("workspace_invites").insert(row).select("*").execute()
    except Exception as exc:
        message = str(exc).lower()
        if "unique" in message or "duplicate" in message:
            raise FileException("An invite is already pending for this email", status_code=409) from exc
        if "foreign key" in message and "invited_by" in message:
            raise FileException("Your account profile is not set up — sign out and sign in again", status_code=409) from exc
        raise FileException("Failed to create invite", status_code=500) from exc

    if not inserted.data:
        raise FileException("Failed to create invite", status_code=500)
    invite_row = inserted.data[0]

    workspace = (
        client.table("workspaces")
        .select("name")
        .eq("id", workspace_id)
        .limit(1)
        .execute()
    )
    workspace_name = workspace.data[0]["name"] if workspace.data else "Study sheet"
    inviter = (
        client.table("profiles")
        .select("email")
        .eq("id", user.id)
        .limit(1)
        .execute()
    )
    inviter_email = inviter.data[0].get("email") if inviter.data else user.email
    email_sent = await send_invite_email(
        to_email=normalized_email,
        workspace_name=workspace_name,
        invite_token=invite_row["token"],
        role=role,
        inviter_email=inviter_email,
    )
    return {**invite_row, "email_sent": email_sent}


async def revoke_workspace_invite(
    client: Client,
    workspace_id: str,
    user: AuthUser,
    *,
    invite_id: str,
) -> None:
    await _check_member_change_rate(user.id)
    require_workspace_role(client, workspace_id, user, min_role="editor")
    invite = (
        client.table("workspace_invites")
        .select("id, accepted_at")
        .eq("id", invite_id)
        .eq("workspace_id", workspace_id)
        .limit(1)
        .execute()
    )
    if not invite.data:
        raise NotFoundException("Invite not found")
    if invite.data[0].get("accepted_at"):
        raise FileException("Cannot revoke an accepted invite", status_code=409)
    client.table("workspace_invites").delete().eq("id", invite_id).eq(
        "workspace_id", workspace_id
    ).execute()


async def remove_workspace_member(
    client: Client,
    workspace_id: str,
    user: AuthUser,
    *,
    member_user_id: str,
) -> None:
    await _check_member_change_rate(user.id)
    require_workspace_role(client, workspace_id, user, min_role="owner")
    if member_user_id == user.id:
        raise FileException("Use leave workspace instead of removing yourself", status_code=400)
    target_role = get_workspace_membership_role(client, workspace_id, AuthUser(id=member_user_id))
    if target_role == "owner":
        raise FileException("Cannot remove the study set owner", status_code=409)
    if target_role is None:
        raise NotFoundException("Member not found")
    client.table("workspace_members").delete().eq("workspace_id", workspace_id).eq(
        "user_id", member_user_id
    ).execute()
    await invalidate_user_workspace_access_caches(
        client,
        user_id=member_user_id,
        workspace_id=workspace_id,
    )


async def leave_workspace(
    client: Client,
    workspace_id: str,
    user: AuthUser,
) -> None:
    await _check_member_change_rate(user.id)
    role = require_workspace_role(client, workspace_id, user, min_role="viewer")
    if role == "owner":
        raise FileException("Owners cannot leave their study set — delete it or transfer ownership first", status_code=409)
    deleted = (
        client.table("workspace_members")
        .delete()
        .eq("workspace_id", workspace_id)
        .eq("user_id", user.id)
        .execute()
    )
    if not deleted.data:
        raise NotFoundException("Membership not found")
    await invalidate_user_workspace_access_caches(
        client,
        user_id=user.id,
        workspace_id=workspace_id,
    )


async def update_member_role(
    client: Client,
    workspace_id: str,
    user: AuthUser,
    *,
    member_user_id: str,
    role: str,
) -> dict:
    await _check_member_change_rate(user.id)
    require_workspace_role(client, workspace_id, user, min_role="owner")
    if role not in {"editor", "viewer"}:
        raise FileException("Role must be editor or viewer")
    if member_user_id == user.id:
        raise FileException("Cannot change your own role", status_code=400)
    target_role = get_workspace_membership_role(client, workspace_id, AuthUser(id=member_user_id))
    if target_role is None:
        raise NotFoundException("Member not found")
    if target_role == "owner":
        raise FileException("Cannot change the owner's role", status_code=409)
    updated = (
        client.table("workspace_members")
        .update({"role": role})
        .eq("workspace_id", workspace_id)
        .eq("user_id", member_user_id)
        .execute()
    )
    if not updated.data:
        raise FileException("Failed to update member role", status_code=500)
    return updated.data[0]


async def accept_workspace_invite(client: Client, user: AuthUser, *, token: str) -> dict:
    await _check_invite_accept_rate(user.id)
    ensure_profile(client, user)
    invite = (
        client.table("workspace_invites")
        .select("*")
        .eq("token", token.strip())
        .limit(1)
        .execute()
    )
    if not invite.data:
        raise NotFoundException("Invite not found or expired")
    row = invite.data[0]
    if row.get("accepted_at"):
        raise FileException("Invite already accepted", status_code=409)
    expires_at = datetime.fromisoformat(row["expires_at"].replace("Z", "+00:00"))
    if expires_at < datetime.now(timezone.utc):
        raise FileException("Invite expired", status_code=410)

    profile = (
        client.table("profiles")
        .select("email")
        .eq("id", user.id)
        .limit(1)
        .execute()
    )
    user_email = (profile.data[0].get("email") if profile.data else None) or user.email or ""
    user_email = user_email.strip().lower()
    if not user_email:
        raise FileException("Your account has no email on file — sign out and sign in again", status_code=409)
    if user_email != row["email"].lower():
        raise FileException("This invite was sent to a different email address", status_code=403)

    client.table("workspace_members").upsert(
        {
            "workspace_id": row["workspace_id"],
            "user_id": user.id,
            "role": row["role"],
            "invited_by": row["invited_by"],
        }
    ).execute()
    client.table("workspace_invites").update(
        {"accepted_at": datetime.now(timezone.utc).isoformat()}
    ).eq("id", row["id"]).execute()

    workspace = (
        client.table("workspaces")
        .select("id, name, description")
        .eq("id", row["workspace_id"])
        .limit(1)
        .execute()
    )
    if not workspace.data:
        raise NotFoundException("Study set not found")
    return {
        "workspace": workspace.data[0],
        "role": row["role"],
    }


async def get_invite_preview(client: Client, token: str) -> dict:
    await _check_invite_preview_rate(token)
    invite = (
        client.table("workspace_invites")
        .select("id, role, expires_at, accepted_at, workspace_id")
        .eq("token", token.strip())
        .limit(1)
        .execute()
    )
    if not invite.data:
        raise NotFoundException("Invite not found")
    row = invite.data[0]
    expires_at = datetime.fromisoformat(row["expires_at"].replace("Z", "+00:00"))
    expired = expires_at < datetime.now(timezone.utc)
    accepted = bool(row.get("accepted_at"))

    if expired or accepted:
        return {
            "valid": False,
            "expired": expired,
            "accepted": accepted,
            "role": row["role"],
        }

    workspace = (
        client.table("workspaces")
        .select("name")
        .eq("id", row["workspace_id"])
        .limit(1)
        .execute()
    )
    return {
        "valid": True,
        "expired": False,
        "accepted": False,
        "role": row["role"],
        "expires_at": row["expires_at"],
        "workspace_name": workspace.data[0]["name"] if workspace.data else None,
    }
