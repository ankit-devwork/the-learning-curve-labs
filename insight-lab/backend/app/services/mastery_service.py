from datetime import datetime, timezone
from typing import Any

from pycorekit.core_logging.logger import get_logger
from supabase import Client

from app.core.auth import AuthUser
from app.core.exceptions import NotFoundException
from app.core.migration_guard import is_missing_phase2_schema, run_or_raise_phase2
from app.core.yaml_config import get_yaml_config

log = get_logger("mastery")


def _get_owned_document(client: Client, document_id: str, user: AuthUser) -> dict:
    result = (
        client.table("documents")
        .select("*")
        .eq("id", document_id)
        .eq("owner_id", user.id)
        .limit(1)
        .execute()
    )
    if not result.data:
        raise NotFoundException("Document not found")
    return result.data[0]


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
    _get_owned_document(client, document_id, user)
    concepts = run_or_raise_phase2(
        lambda: client.table("document_concepts")
        .select("concept_id, name, topic")
        .eq("document_id", document_id)
        .order("name")
        .execute()
        .data
        or []
    )
    mastery_rows = run_or_raise_phase2(
        lambda: client.table("concept_mastery")
        .select("concept_id, attempts, correct, last_attempt_at")
        .eq("document_id", document_id)
        .eq("user_id", user.id)
        .execute()
        .data
        or []
    )
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
