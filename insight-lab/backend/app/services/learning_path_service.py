"""Learning paths — prerequisite-aware concept ordering for a study set."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any
from uuid import uuid4

from pycorekit.exceptions.file import FileException
from supabase import Client

from app.core.auth import AuthUser
from app.core.migration_guard import PHASE2_MIGRATION_NOTICE, is_missing_phase2_schema
from app.services.mastery_service import get_workspace_concept_mastery
from app.services.workspace_access import require_workspace_role


def _topological_concept_order(
    concepts: list[dict[str, Any]],
    relationships: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_id = {row["concept_id"]: row for row in concepts}
    in_degree: dict[str, int] = {cid: 0 for cid in by_id}
    dependents: dict[str, list[str]] = defaultdict(list)

    for rel in relationships:
        if rel.get("relationship_type") != "prerequisite_for":
            continue
        source = rel.get("source_concept_id")
        target = rel.get("target_concept_id")
        if source not in by_id or target not in by_id or source == target:
            continue
        dependents[source].append(target)
        in_degree[target] += 1

    queue = deque(sorted(cid for cid, degree in in_degree.items() if degree == 0))
    ordered_ids: list[str] = []
    while queue:
        node = queue.popleft()
        ordered_ids.append(node)
        for neighbor in sorted(dependents.get(node, [])):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(ordered_ids) < len(by_id):
        remaining = sorted(cid for cid in by_id if cid not in ordered_ids)
        ordered_ids.extend(remaining)

    return [by_id[cid] for cid in ordered_ids if cid in by_id]


async def generate_workspace_learning_path(
    client: Client,
    workspace_id: str,
    user: AuthUser,
) -> dict[str, Any]:
    require_workspace_role(client, workspace_id, user, min_role="viewer")

    docs = (
        client.table("documents")
        .select("id, filename")
        .eq("workspace_id", workspace_id)
        .eq("file_type", "document")
        .eq("status", "ready")
        .order("created_at")
        .execute()
        .data
        or []
    )
    if not docs:
        raise FileException(
            "No ready documents — upload and process PDFs or Word files first",
            status_code=409,
        )

    mastery_payload = await get_workspace_concept_mastery(client, workspace_id, user)
    mastery_by_key = {
        (row.get("document_id"), row.get("concept_id")): row
        for row in mastery_payload.get("concepts") or []
    }

    path_nodes: list[dict[str, Any]] = []
    sort_order = 0

    try:
        for doc in docs:
            document_id = doc["id"]
            concepts = (
                client.table("document_concepts")
                .select("concept_id, name, topic")
                .eq("document_id", document_id)
                .order("name")
                .execute()
                .data
                or []
            )
            if not concepts:
                continue

            relationships = (
                client.table("concept_relationships")
                .select("source_concept_id, target_concept_id, relationship_type")
                .eq("document_id", document_id)
                .execute()
                .data
                or []
            )
            ordered = _topological_concept_order(concepts, relationships)

            def mastery_rank(concept: dict[str, Any]) -> tuple[int, float]:
                mastery = mastery_by_key.get((document_id, concept["concept_id"]), {})
                attempts = int(mastery.get("attempts") or 0)
                percent = float(mastery.get("percent") or 100)
                if attempts == 0:
                    return (0, 0)
                if percent < 60:
                    return (1, percent)
                return (2, -percent)

            ordered.sort(key=mastery_rank)

            for concept in ordered:
                mastery = mastery_by_key.get((document_id, concept["concept_id"]), {})
                attempts = int(mastery.get("attempts") or 0)
                percent = mastery.get("percent")
                if attempts == 0:
                    status = "available"
                elif percent is not None and percent < 60:
                    status = "needs_practice"
                else:
                    status = "completed"

                path_nodes.append(
                    {
                        "sort_order": sort_order,
                        "node_kind": "concept",
                        "document_id": document_id,
                        "document_filename": doc["filename"],
                        "concept_id": concept["concept_id"],
                        "concept_name": concept["name"],
                        "topic": concept.get("topic"),
                        "status": status,
                        "mastery_percent": percent,
                    }
                )
                sort_order += 1
    except Exception as exc:
        if is_missing_phase2_schema(exc):
            return {
                "workspace_id": workspace_id,
                "path_id": None,
                "title": "Learning path unavailable",
                "nodes": [],
                "migration_required": True,
                "notice": PHASE2_MIGRATION_NOTICE,
            }
        raise

    if not path_nodes:
        raise FileException(
            "No concepts found yet — concepts are extracted when documents finish processing",
            status_code=409,
        )

    path_id = str(uuid4())
    title = f"Learning path — {len(docs)} document{'s' if len(docs) != 1 else ''}"
    client.table("learning_paths").insert(
        {
            "id": path_id,
            "workspace_id": workspace_id,
            "created_by": user.id,
            "title": title,
            "path_type": "generated",
        }
    ).execute()

    node_rows = []
    response_nodes = []
    for node in path_nodes:
        node_id = str(uuid4())
        node_rows.append(
            {
                "id": node_id,
                "path_id": path_id,
                "sort_order": node["sort_order"],
                "node_kind": node["node_kind"],
                "document_id": node["document_id"],
                "concept_id": node["concept_id"],
                "concept_name": node["concept_name"],
                "topic": node.get("topic"),
                "metadata": {"status": node["status"], "mastery_percent": node.get("mastery_percent")},
            }
        )
        response_nodes.append({**node, "id": node_id, "path_id": path_id})

    client.table("learning_path_nodes").insert(node_rows).execute()

    return {
        "workspace_id": workspace_id,
        "path_id": path_id,
        "title": title,
        "node_count": len(response_nodes),
        "nodes": response_nodes,
    }


def get_latest_workspace_learning_path(
    client: Client,
    workspace_id: str,
    user: AuthUser,
) -> dict[str, Any] | None:
    require_workspace_role(client, workspace_id, user, min_role="viewer")
    path_row = (
        client.table("learning_paths")
        .select("id, title, created_at")
        .eq("workspace_id", workspace_id)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
        .data
    )
    if not path_row:
        return None

    path = path_row[0]
    nodes = (
        client.table("learning_path_nodes")
        .select("id, sort_order, node_kind, document_id, concept_id, concept_name, topic, metadata")
        .eq("path_id", path["id"])
        .order("sort_order")
        .execute()
        .data
        or []
    )

    doc_names: dict[str, str] = {}
    doc_ids = {row["document_id"] for row in nodes if row.get("document_id")}
    if doc_ids:
        docs = (
            client.table("documents")
            .select("id, filename")
            .in_("id", list(doc_ids))
            .execute()
            .data
            or []
        )
        doc_names = {row["id"]: row["filename"] for row in docs}

    response_nodes = []
    for row in nodes:
        metadata = row.get("metadata") or {}
        response_nodes.append(
            {
                "id": row["id"],
                "sort_order": row["sort_order"],
                "node_kind": row["node_kind"],
                "document_id": row.get("document_id"),
                "document_filename": doc_names.get(row.get("document_id") or "", ""),
                "concept_id": row.get("concept_id"),
                "concept_name": row.get("concept_name"),
                "topic": row.get("topic"),
                "status": metadata.get("status", "available"),
                "mastery_percent": metadata.get("mastery_percent"),
            }
        )

    return {
        "workspace_id": workspace_id,
        "path_id": path["id"],
        "title": path["title"],
        "created_at": path["created_at"],
        "node_count": len(response_nodes),
        "nodes": response_nodes,
    }
