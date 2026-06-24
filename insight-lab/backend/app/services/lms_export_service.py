"""LMS export bundles — Common Cartridge (.imscc) and zip resource pack."""

from __future__ import annotations

import io
import re
import zipfile
from typing import Any

from pycorekit.exceptions.file import FileException
from supabase import Client

from app.core.auth import AuthUser
from app.services.export_utils import (
    build_imsmanifest_xml,
    course_pack_to_markdown,
    flashcards_to_anki_csv,
    quiz_to_qti_xml,
    study_guide_to_markdown,
)
from app.services.workspace_access import require_workspace_role
from app.services.workspace_service import get_workspace_row, list_workspace_documents


def _safe_slug(text: str, *, max_len: int = 48) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", text.strip().lower()).strip("-")
    return (slug or "item")[:max_len]


def collect_ready_material_summaries(
    client: Client,
    workspace_id: str,
    user: AuthUser,
    *,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Summaries for ready PDF/Word docs and analyzed spreadsheets."""
    documents = list_workspace_documents(client, workspace_id, user, limit=limit)
    summaries: list[dict[str, Any]] = []

    for doc in documents:
        if doc.get("status") != "ready":
            continue
        file_type = doc.get("file_type")
        if file_type not in ("document", "excel"):
            continue

        detail = (
            client.table("documents")
            .select("summary, excel_summary")
            .eq("id", doc["id"])
            .limit(1)
            .execute()
            .data
            or []
        )
        row = detail[0] if detail else {}
        summary = row.get("summary") if file_type == "document" else row.get("excel_summary")
        summaries.append(
            {
                "filename": doc.get("filename"),
                "summary": summary,
                "file_type": file_type,
            }
        )

    return summaries


def _has_exportable_content(*, summaries: list[dict[str, Any]], items: list[dict[str, Any]]) -> bool:
    if items:
        return True
    return any(entry.get("summary") for entry in summaries)


def _append_document_export_items(
    client: Client,
    doc: dict[str, Any],
    items: list[dict[str, Any]],
) -> None:
    doc_id = doc["id"]
    filename = doc.get("filename") or "document"
    slug = _safe_slug(filename)

    detail = (
        client.table("documents").select("summary").eq("id", doc_id).limit(1).execute().data or []
    )
    summary = detail[0].get("summary") if detail else None

    quiz_row = (
        client.table("quizzes")
        .select("id, title")
        .eq("document_id", doc_id)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
        .data
    )
    if quiz_row:
        quiz_id = quiz_row[0]["id"]
        questions = (
            client.table("quiz_questions")
            .select("question_text, options, correct_option_index")
            .eq("quiz_id", quiz_id)
            .order("sort_order")
            .execute()
            .data
            or []
        )
        if questions:
            items.append(
                {
                    "type": "quiz",
                    "slug": slug,
                    "title": quiz_row[0].get("title") or f"Quiz — {filename}",
                    "filename": filename,
                    "qti_xml": quiz_to_qti_xml(
                        title=quiz_row[0].get("title") or filename,
                        questions=questions,
                    ),
                }
            )

    guide_row = (
        client.table("study_guides")
        .select("id, title, content")
        .eq("document_id", doc_id)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
        .data
    )
    if guide_row:
        guide = guide_row[0]
        items.append(
            {
                "type": "study_guide",
                "slug": slug,
                "title": guide.get("title") or f"Study guide — {filename}",
                "markdown": study_guide_to_markdown(title=guide["title"], content=guide["content"]),
            }
        )

    fc_set = (
        client.table("flashcard_sets")
        .select("id, title")
        .eq("document_id", doc_id)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
        .data
    )
    if fc_set:
        cards = (
            client.table("flashcards")
            .select("front, back")
            .eq("set_id", fc_set[0]["id"])
            .order("sort_order")
            .execute()
            .data
            or []
        )
        if cards:
            items.append(
                {
                    "type": "flashcards",
                    "slug": slug,
                    "title": fc_set[0].get("title") or f"Flashcards — {filename}",
                    "anki_csv": flashcards_to_anki_csv(cards),
                }
            )

    if summary:
        items.append(
            {
                "type": "summary",
                "slug": slug,
                "title": f"Summary — {filename}",
                "markdown": f"# {filename}\n\n{summary}\n",
            }
        )


def _append_excel_export_items(
    client: Client,
    doc: dict[str, Any],
    items: list[dict[str, Any]],
) -> None:
    doc_id = doc["id"]
    filename = doc.get("filename") or "spreadsheet"
    slug = _safe_slug(filename)

    detail = (
        client.table("documents")
        .select("excel_summary")
        .eq("id", doc_id)
        .limit(1)
        .execute()
        .data
        or []
    )
    summary = detail[0].get("excel_summary") if detail else None
    if not summary:
        return

    items.append(
        {
            "type": "summary",
            "slug": slug,
            "title": f"Spreadsheet summary — {filename}",
            "markdown": f"# {filename} (spreadsheet)\n\n{summary}\n",
        }
    )


def _collect_workspace_export_items(
    client: Client, workspace_id: str, user: AuthUser
) -> dict[str, Any]:
    documents = list_workspace_documents(client, workspace_id, user, limit=100)
    items: list[dict[str, Any]] = []

    for doc in documents:
        if doc.get("status") != "ready":
            continue
        file_type = doc.get("file_type")
        if file_type == "document":
            _append_document_export_items(client, doc, items)
        elif file_type == "excel":
            _append_excel_export_items(client, doc, items)

    return {"documents": documents, "items": items}


def build_lms_bundle_zip(
    client: Client,
    workspace_id: str,
    user: AuthUser,
    *,
    include_manifest: bool = True,
) -> tuple[bytes, str]:
    """Return zip bytes and suggested filename (.imscc if manifest included)."""
    require_workspace_role(client, workspace_id, user, min_role="editor")
    workspace = get_workspace_row(client, workspace_id, user)
    workspace_name = workspace.get("name") or "study-sheet"
    collected = _collect_workspace_export_items(client, workspace_id, user)
    items = collected["items"]
    summaries = collect_ready_material_summaries(client, workspace_id, user)

    if not _has_exportable_content(summaries=summaries, items=items):
        raise FileException(
            "Nothing to export yet — add ready PDFs/Word docs or analyzed spreadsheets with summaries",
            status_code=409,
        )

    buffer = io.BytesIO()
    manifest_resources: list[dict[str, str]] = []

    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        readme = f"""# InsightLab LMS export — {workspace_name}

## Canvas / Moodle import
1. Download this package ({'*.imscc' if include_manifest else '*.zip'}).
2. In Canvas: Settings → Import course content → Common Cartridge 1.x.
3. Or import individual QTI files from the assessments/ folder.

## Contents
- assessments/*.xml — QTI 1.2 quizzes
- study-guides/*.md — study guides
- flashcards/*.csv — Anki-compatible flashcards
- summaries/*.md — document and spreadsheet summaries
- course-overview.md — combined overview
"""
        archive.writestr("README-CANVAS-IMPORT.md", readme)
        archive.writestr(
            "course-overview.md",
            course_pack_to_markdown(workspace_name=workspace_name, documents=summaries),
        )

        for index, item in enumerate(items, start=1):
            ident = f"res{index}"
            if item["type"] == "quiz":
                path = f"assessments/{item['slug']}-quiz.xml"
                archive.writestr(path, item["qti_xml"])
                manifest_resources.append(
                    {
                        "identifier": ident,
                        "title": item["title"],
                        "type": "imsqti_xmlv1p2",
                        "href": path,
                    }
                )
            elif item["type"] == "study_guide":
                path = f"study-guides/{item['slug']}-guide.md"
                archive.writestr(path, item["markdown"])
                manifest_resources.append(
                    {
                        "identifier": ident,
                        "title": item["title"],
                        "type": "webcontent",
                        "href": path,
                    }
                )
            elif item["type"] == "flashcards":
                path = f"flashcards/{item['slug']}-cards.csv"
                archive.writestr(path, item["anki_csv"])
                manifest_resources.append(
                    {
                        "identifier": ident,
                        "title": item["title"],
                        "type": "webcontent",
                        "href": path,
                    }
                )
            elif item["type"] == "summary":
                path = f"summaries/{item['slug']}-summary.md"
                archive.writestr(path, item["markdown"])
                manifest_resources.append(
                    {
                        "identifier": ident,
                        "title": item["title"],
                        "type": "webcontent",
                        "href": path,
                    }
                )

        if include_manifest:
            archive.writestr(
                "imsmanifest.xml",
                build_imsmanifest_xml(title=workspace_name, resources=manifest_resources),
            )

    slug = _safe_slug(workspace_name)
    filename = f"{slug}-canvas.imscc" if include_manifest else f"{slug}-lms-bundle.zip"
    return buffer.getvalue(), filename
