"""Transactional email delivery (Resend HTTP API)."""

from __future__ import annotations

import os

import httpx

from pycorekit.core_logging.logger import get_logger

from app.core.yaml_config import get_yaml_config

log = get_logger("email")


def _frontend_base_url() -> str:
    cfg = get_yaml_config().email
    return (os.getenv("FRONTEND_BASE_URL") or cfg.frontend_base_url or "http://localhost:3000").rstrip("/")


async def send_invite_email(
    *,
    to_email: str,
    workspace_name: str,
    invite_token: str,
    role: str,
    inviter_email: str | None = None,
) -> bool:
    cfg = get_yaml_config().email
    if not cfg.enabled:
        return False

    api_key = os.getenv(cfg.resend_api_key_env or "RESEND_API_KEY", "").strip()
    if not api_key:
        log.warning("Invite email skipped — API key not configured")
        return False

    invite_url = f"{_frontend_base_url()}/invite/{invite_token}"
    inviter_line = f"{inviter_email} invited you" if inviter_email else "You have been invited"
    subject = f"Join \"{workspace_name}\" on InsightLab"
    html = f"""
    <p>{inviter_line} to collaborate on the study sheet <strong>{workspace_name}</strong>
    as a <strong>{role}</strong>.</p>
    <p><a href="{invite_url}">Accept invite</a></p>
    <p style="color:#666;font-size:12px">Or copy this link: {invite_url}</p>
    """

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                "https://api.resend.com/emails",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "from": cfg.from_address,
                    "to": [to_email],
                    "subject": subject,
                    "html": html,
                },
            )
        if response.status_code >= 400:
            log.warning("Invite email failed", status=response.status_code, body=response.text[:200])
            return False
        log.info("Invite email sent", to=to_email, workspace=workspace_name)
        return True
    except Exception as exc:
        log.warning("Invite email error", error=str(exc))
        return False


async def send_artifact_ready_email(
    *,
    to_email: str,
    workspace_name: str,
    artifact_label: str,
    link_url: str,
) -> bool:
    cfg = get_yaml_config().email
    if not cfg.enabled or not cfg.notify_artifact_ready:
        return False

    api_key = os.getenv(cfg.resend_api_key_env or "RESEND_API_KEY", "").strip()
    if not api_key:
        return False

    subject = f"{artifact_label} ready — {workspace_name}"
    html = f"""
    <p>Your <strong>{artifact_label}</strong> for study sheet
    <strong>{workspace_name}</strong> is ready.</p>
    <p><a href="{link_url}">Open in InsightLab</a></p>
    """

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                "https://api.resend.com/emails",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "from": cfg.from_address,
                    "to": [to_email],
                    "subject": subject,
                    "html": html,
                },
            )
        return response.status_code < 400
    except Exception:
        return False
