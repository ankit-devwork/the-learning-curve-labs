"""Pure helpers for study session plan shaping (no I/O dependencies)."""

from __future__ import annotations

from typing import Any


def apply_learning_path_order(plan: dict[str, Any], path_nodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Reorder document-scoped session steps to match a generated learning path."""
    if not path_nodes:
        return plan

    doc_order: list[str] = []
    seen: set[str] = set()
    for node in sorted(path_nodes, key=lambda row: int(row.get("sort_order") or 0)):
        document_id = node.get("document_id")
        if document_id and document_id not in seen:
            seen.add(document_id)
            doc_order.append(document_id)

    if not doc_order:
        return plan

    steps = plan.get("steps") or []
    focus_steps = [step for step in steps if step.get("step") == "focus"]
    tail_steps = [step for step in steps if step.get("step") in ("adaptive_quiz", "set_quiz")]
    doc_steps = [step for step in steps if step.get("step") in ("brief", "flashcards")]

    grouped: dict[str, list[dict[str, Any]]] = {}
    for step in doc_steps:
        document_id = step.get("document_id")
        if not document_id:
            continue
        grouped.setdefault(document_id, []).append(step)

    ordered_doc_steps: list[dict[str, Any]] = []
    for document_id in doc_order:
        ordered_doc_steps.extend(grouped.pop(document_id, []))
    for leftover in grouped.values():
        ordered_doc_steps.extend(leftover)

    reordered = focus_steps + ordered_doc_steps + tail_steps
    return {
        **plan,
        "steps": reordered,
        "path_document_order": doc_order,
        "estimated_minutes": sum(int(step.get("duration_min") or 0) for step in reordered),
    }
