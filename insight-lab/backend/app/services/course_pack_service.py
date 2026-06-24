"""Generate a bundled course pack for all ready documents in a study set."""

from __future__ import annotations

from typing import Any

from pycorekit.core_logging.logger import get_logger
from pycorekit.exceptions.file import FileException
from supabase import Client

from app.core.auth import AuthUser
from app.core.cache import check_rate_limit
from app.core.exceptions import RateLimitException
from app.core.yaml_config import get_yaml_config
from app.services.artifact_service import generate_document_flashcards, generate_document_study_guide
from app.services.audio_overview_service import generate_audio_overview
from app.services.document_service import get_document_summary
from app.services.quiz_service import generate_document_quiz
from app.services.workspace_access import require_workspace_role

log = get_logger("course_pack")


async def _build_document_pack_item(
    client: Client,
    doc: dict[str, Any],
    user: AuthUser,
) -> dict[str, Any]:
    document_id = doc["id"]
    item: dict[str, Any] = {
        "document_id": document_id,
        "filename": doc["filename"],
        "file_type": "document",
        "artifacts": {},
        "errors": [],
    }
    try:
        summary = await get_document_summary(client, document_id, user)
        item["artifacts"]["summary"] = summary.get("summary")
    except Exception as exc:
        item["errors"].append(f"summary: {exc}")

    for name, factory in (
        ("quiz", lambda: generate_document_quiz(
            client, document_id, user,
            question_type="scq", difficulty="medium", num_questions=5,
        )),
        ("flashcards", lambda: generate_document_flashcards(
            client, document_id, user, num_cards=10,
        )),
        ("study_guide", lambda: generate_document_study_guide(client, document_id, user)),
        ("audio_overview", lambda: generate_audio_overview(client, document_id, user)),
    ):
        try:
            payload = await factory()
            if name == "quiz":
                item["artifacts"]["quiz_id"] = payload.get("quiz_id")
            elif name == "flashcards":
                item["artifacts"]["flashcard_set_id"] = payload.get("set_id")
            elif name == "study_guide":
                item["artifacts"]["study_guide_id"] = payload.get("guide_id")
            elif name == "audio_overview":
                item["artifacts"]["audio_title"] = payload.get("title")
                item["artifacts"]["audio_script"] = payload.get("script")
        except Exception as exc:
            item["errors"].append(f"{name}: {exc}")

    return item


def _build_excel_pack_item(client: Client, doc: dict[str, Any]) -> dict[str, Any]:
    document_id = doc["id"]
    item: dict[str, Any] = {
        "document_id": document_id,
        "filename": doc["filename"],
        "file_type": "excel",
        "artifacts": {},
        "errors": [],
    }
    detail = (
        client.table("documents")
        .select("excel_summary, excel_charts")
        .eq("id", document_id)
        .limit(1)
        .execute()
        .data
        or []
    )
    row = detail[0] if detail else {}
    summary = row.get("excel_summary")
    if summary:
        item["artifacts"]["summary"] = summary
        charts = row.get("excel_charts") or []
        if charts:
            item["artifacts"]["chart_count"] = len(charts)
    else:
        item["errors"].append(
            "summary: Spreadsheet is ready but has no analysis summary yet — open it and run Analyze first",
        )
    return item


async def generate_course_pack(
    client: Client,
    workspace_id: str,
    user: AuthUser,
    *,
    document_ids: list[str] | None = None,
) -> dict[str, Any]:
    cfg = get_yaml_config().course_pack
    allowed, retry_after = await check_rate_limit(
        key=f"course_pack:{user.id}",
        limit=cfg.generate_rate_limit_per_hour,
        window_seconds=3600,
    )
    if not allowed:
        raise RateLimitException(
            f"Course pack limit reached ({cfg.generate_rate_limit_per_hour}/hour)",
            retry_after=retry_after,
        )

    require_workspace_role(client, workspace_id, user, min_role="editor")

    query = (
        client.table("documents")
        .select("id, filename, file_type, status, workspace_id")
        .eq("workspace_id", workspace_id)
        .eq("status", "ready")
        .in_("file_type", ["document", "excel"])
    )
    if document_ids:
        query = query.in_("id", document_ids)
    docs = query.order("created_at").execute().data or []
    if not docs:
        raise FileException(
            "No ready materials found in this study set — upload and process PDFs, Word docs, or spreadsheets first",
            status_code=409,
        )

    results: list[dict[str, Any]] = []
    for doc in docs:
        if doc["file_type"] == "excel":
            results.append(_build_excel_pack_item(client, doc))
        else:
            results.append(await _build_document_pack_item(client, doc, user))

    log.info(
        "Course pack generated",
        workspace_id=workspace_id,
        user_id=user.id,
        document_count=len(results),
    )
    return {
        "workspace_id": workspace_id,
        "document_count": len(results),
        "documents": results,
    }
