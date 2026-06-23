from datetime import datetime, timezone
from typing import Any

from pycorekit.core_logging.logger import get_logger
from supabase import Client

from app.core.auth import AuthUser
from app.core.exceptions import NotFoundException
from app.core.migration_guard import PHASE2_MIGRATION_NOTICE, is_missing_phase2_schema, run_or_none_phase2, run_or_raise_phase2
from app.core.yaml_config import get_yaml_config
from app.services.workspace_access import get_accessible_document, require_workspace_role

log = get_logger("mastery")


def record_quiz_mastery(
    client: Client,
    *,
    user: AuthUser,
    document_id: str,
    questions: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> None:
    try:
        _record_quiz_mastery_rows(
            client,
            user=user,
            document_id=document_id,
            questions=questions,
            results=results,
        )
    except Exception as exc:
        if is_missing_phase2_schema(exc):
            log.warning(
                "Skipping concept mastery update; Phase 2 migration not applied",
                document_id=document_id,
                user_id=user.id,
            )
            return
        raise


def _record_quiz_mastery_rows(
    client: Client,
    *,
    user: AuthUser,
    document_id: str,
    questions: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> None:
    result_by_question = {row["question_id"]: row for row in results}
    now = datetime.now(timezone.utc).isoformat()

    for question in questions:
        concept_id = question.get("concept_id")
        if not concept_id:
            continue
        result = result_by_question.get(question["id"])
        if not result:
            continue

        existing = run_or_raise_phase2(
            lambda: client.table("concept_mastery")
            .select("attempts, correct")
            .eq("user_id", user.id)
            .eq("document_id", document_id)
            .eq("concept_id", concept_id)
            .limit(1)
            .execute()
        )
        if existing.data:
            row = existing.data[0]
            attempts = int(row["attempts"]) + 1
            correct = int(row["correct"]) + (1 if result["correct"] else 0)
            run_or_raise_phase2(
                lambda: client.table("concept_mastery")
                .update(
                    {
                        "attempts": attempts,
                        "correct": correct,
                        "last_attempt_at": now,
                    }
                )
                .eq("user_id", user.id)
                .eq("document_id", document_id)
                .eq("concept_id", concept_id)
                .execute()
            )
        else:
            run_or_raise_phase2(
                lambda: client.table("concept_mastery")
                .insert(
                    {
                        "user_id": user.id,
                        "document_id": document_id,
                        "concept_id": concept_id,
                        "attempts": 1,
                        "correct": 1 if result["correct"] else 0,
                        "last_attempt_at": now,
                    }
                )
                .execute()
            )


async def get_concept_mastery(
    client: Client,
    document_id: str,
    user: AuthUser,
) -> dict[str, Any]:
    get_accessible_document(client, document_id, user, min_role="viewer")
    try:
        concepts = run_or_none_phase2(
            lambda: client.table("document_concepts")
            .select("concept_id, name, topic")
            .eq("document_id", document_id)
            .order("name")
            .execute()
            .data
            or []
        )
        if concepts is None:
            return {
                "document_id": document_id,
                "concepts": [],
                "migration_required": True,
                "notice": PHASE2_MIGRATION_NOTICE,
            }

        mastery_rows = run_or_none_phase2(
            lambda: client.table("concept_mastery")
            .select("concept_id, attempts, correct, last_attempt_at")
            .eq("document_id", document_id)
            .eq("user_id", user.id)
            .execute()
            .data
            or []
        )
        if mastery_rows is None:
            return {
                "document_id": document_id,
                "concepts": [],
                "migration_required": True,
                "notice": PHASE2_MIGRATION_NOTICE,
            }
    except Exception as exc:
        if is_missing_phase2_schema(exc):
            return {
                "document_id": document_id,
                "concepts": [],
                "migration_required": True,
                "notice": PHASE2_MIGRATION_NOTICE,
            }
        raise

    mastery_map = {row["concept_id"]: row for row in mastery_rows}

    items: list[dict[str, Any]] = []
    for concept in concepts:
        concept_id = concept["concept_id"]
        mastery = mastery_map.get(concept_id)
        attempts = int(mastery["attempts"]) if mastery else 0
        correct = int(mastery["correct"]) if mastery else 0
        percent = round((correct / attempts) * 100, 1) if attempts else None
        items.append(
            {
                "concept_id": concept_id,
                "name": concept["name"],
                "topic": concept.get("topic"),
                "attempts": attempts,
                "correct": correct,
                "percent": percent,
                "last_attempt_at": mastery.get("last_attempt_at") if mastery else None,
            }
        )

    return {"document_id": document_id, "concepts": items}


async def get_weak_concepts(
    client: Client,
    document_id: str,
    user: AuthUser,
) -> list[dict[str, Any]]:
    cfg = get_yaml_config().adaptive_quiz
    mastery = await get_concept_mastery(client, document_id, user)
    weak: list[dict[str, Any]] = []
    for item in mastery["concepts"]:
        attempts = item["attempts"]
        if attempts < cfg.min_attempts_before_adaptive:
            continue
        percent = item.get("percent")
        if percent is not None and percent < cfg.weak_threshold_percent:
            weak.append(item)

    weak.sort(key=lambda row: (row.get("percent") or 0, -row["attempts"]))
    return weak[: cfg.max_weak_concepts]


async def get_workspace_concept_mastery(
    client: Client,
    workspace_id: str,
    user: AuthUser,
) -> dict[str, Any]:
    from app.services.workspace_access import require_workspace_role

    require_workspace_role(client, workspace_id, user, min_role="viewer")
    docs = (
        client.table("documents")
        .select("id, filename")
        .eq("workspace_id", workspace_id)
        .eq("file_type", "document")
        .eq("status", "ready")
        .execute()
        .data
        or []
    )

    all_concepts: list[dict[str, Any]] = []
    for doc in docs:
        mastery = await get_concept_mastery(client, doc["id"], user)
        for item in mastery.get("concepts") or []:
            all_concepts.append(
                {
                    **item,
                    "document_id": doc["id"],
                    "document_filename": doc["filename"],
                }
            )

    return {
        "workspace_id": workspace_id,
        "concepts": all_concepts,
        "document_count": len(docs),
    }


async def get_workspace_weak_concepts(
    client: Client,
    workspace_id: str,
    user: AuthUser,
) -> list[dict[str, Any]]:
    cfg = get_yaml_config().adaptive_quiz
    mastery = await get_workspace_concept_mastery(client, workspace_id, user)
    weak: list[dict[str, Any]] = []
    for item in mastery["concepts"]:
        attempts = item["attempts"]
        if attempts < cfg.min_attempts_before_adaptive:
            continue
        percent = item.get("percent")
        if percent is not None and percent < cfg.weak_threshold_percent:
            weak.append(item)

    weak.sort(key=lambda row: (row.get("percent") or 0, -row["attempts"]))
    return weak[: cfg.max_weak_concepts]
