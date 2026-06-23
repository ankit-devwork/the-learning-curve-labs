from datetime import datetime, timezone
from typing import Any

from pycorekit.core_logging.logger import get_logger
from pycorekit.exceptions.file import FileException
from supabase import Client

from app.core.auth import AuthUser
from app.core.cache import cache_get, cache_set, check_rate_limit
from app.core.exceptions import NotFoundException, RateLimitException
from app.core.migration_guard import (
    PHASE2_MIGRATION_NOTICE,
    is_missing_phase2_schema,
    run_or_raise_phase2,
)
from app.core.neo4j_client import neo4j_client
from app.core.yaml_config import get_yaml_config
from app.services.concept_extraction import draft_to_rows, parse_concept_extraction
from app.services.llm_client import extract_concepts_from_chunks, graph_cache_key
from app.services.workspace_access import get_accessible_document, require_editable_document

log = get_logger("graph")


def _sample_chunks(chunks: list[dict], max_chunks: int) -> list[dict]:
    if len(chunks) <= max_chunks:
        return chunks
    step = len(chunks) / max_chunks
    indices = sorted({min(int(index * step), len(chunks) - 1) for index in range(max_chunks)})
    return [chunks[index] for index in indices]


async def _sync_to_neo4j(
    *,
    user_id: str,
    document_id: str,
    workspace_id: str,
    filename: str,
    concepts: list[dict[str, Any]],
    relationships: list[dict[str, Any]],
) -> bool:
    if not neo4j_client.is_configured:
        return False

    try:
        driver = neo4j_client.get_driver()
        async with driver.session() as session:
            await session.run(
                """
                MERGE (u:User {id: $user_id})
                MERGE (w:Workspace {id: $workspace_id})
                MERGE (u)-[:OWNS]->(w)
                MERGE (d:Document {id: $document_id})
                SET d.filename = $filename
                MERGE (w)-[:HAS]->(d)
                WITH d
                OPTIONAL MATCH (d)-[:MENTIONS]->(c:Concept)
                DETACH DELETE c
                """,
                user_id=user_id,
                workspace_id=workspace_id,
                document_id=document_id,
                filename=filename,
            )
            await session.run(
                """
                UNWIND $concepts AS concept
                MERGE (c:Concept {id: concept.concept_id, document_id: $document_id})
                SET c.name = concept.name,
                    c.topic = concept.topic,
                    c.chunk_indexes = concept.chunk_indexes
                WITH c
                MATCH (d:Document {id: $document_id})
                MERGE (d)-[:MENTIONS]->(c)
                """,
                document_id=document_id,
                concepts=[
                    {
                        "concept_id": row["concept_id"],
                        "name": row["name"],
                        "topic": row.get("topic"),
                        "chunk_indexes": row.get("chunk_indexes") or [],
                    }
                    for row in concepts
                ],
            )
            if relationships:
                await session.run(
                    """
                    UNWIND $relationships AS rel
                    MATCH (source:Concept {id: rel.source_concept_id, document_id: $document_id})
                    MATCH (target:Concept {id: rel.target_concept_id, document_id: $document_id})
                    MERGE (source)-[r:RELATED_TO]->(target)
                    SET r.type = rel.relationship_type
                    """,
                    document_id=document_id,
                    relationships=relationships,
                )
        return True
    except Exception as exc:
        log.warning(
            "Neo4j sync failed; Postgres graph remains available",
            document_id=document_id,
            error=str(exc),
        )
        return False


async def sync_document_graph(
    client: Client,
    document_id: str,
    user: AuthUser,
    *,
    skip_rate_limit: bool = False,
) -> dict[str, Any]:
    cfg = get_yaml_config().graph
    if not skip_rate_limit:
        allowed, retry_after = await check_rate_limit(
            key=f"graph_sync:{user.id}",
            limit=cfg.sync_rate_limit_per_hour,
            window_seconds=3600,
        )
        if not allowed:
            raise RateLimitException(
                f"Graph sync rate limit reached ({cfg.sync_rate_limit_per_hour}/hour)",
                retry_after=retry_after,
            )

    doc = require_editable_document(client, document_id, user)
    if doc["file_type"] != "document":
        raise FileException("Concept graphs are only available for document uploads")
    if doc["status"] != "ready":
        raise FileException("Document must be processed before syncing concepts", status_code=409)

    cache_key = graph_cache_key(user.id, document_id)
    cached = await cache_get(cache_key)
    if cached and doc.get("neo4j_synced_at"):
        return {**cached, "cached": True}

    chunks_result = (
        client.table("document_chunks")
        .select("chunk_index, content")
        .eq("document_id", document_id)
        .order("chunk_index")
        .execute()
    )
    chunks = chunks_result.data or []
    if not chunks:
        raise FileException("No document chunks found; reprocess the document", status_code=409)

    sampled = _sample_chunks(chunks, get_yaml_config().quizzes.max_context_chunks)
    context_chunks = [row["content"] for row in sampled]
    chunk_indexes = [row["chunk_index"] for row in sampled]

    raw = await extract_concepts_from_chunks(
        context_chunks=context_chunks,
        chunk_indexes=chunk_indexes,
        filename=doc["filename"],
    )
    try:
        draft = parse_concept_extraction(raw, max_concepts=cfg.max_concepts_per_document)
    except (ValueError, Exception) as exc:
        raise FileException(f"Invalid concept payload from LLM: {exc}", status_code=502) from exc

    concept_rows, relationship_rows = draft_to_rows(draft, document_id=document_id)

    run_or_raise_phase2(
        lambda: client.table("document_concepts").delete().eq("document_id", document_id).execute()
    )
    run_or_raise_phase2(
        lambda: client.table("concept_relationships").delete().eq("document_id", document_id).execute()
    )
    if concept_rows:
        run_or_raise_phase2(lambda: client.table("document_concepts").insert(concept_rows).execute())
    if relationship_rows:
        run_or_raise_phase2(
            lambda: client.table("concept_relationships").insert(relationship_rows).execute()
        )

    neo4j_synced = await _sync_to_neo4j(
        user_id=user.id,
        document_id=document_id,
        workspace_id=doc["workspace_id"],
        filename=doc["filename"],
        concepts=concept_rows,
        relationships=relationship_rows,
    )

    synced_at = datetime.now(timezone.utc).isoformat()
    client.table("documents").update({"neo4j_synced_at": synced_at}).eq("id", document_id).execute()

    payload = {
        "document_id": document_id,
        "concept_count": len(concept_rows),
        "relationship_count": len(relationship_rows),
        "neo4j_synced": neo4j_synced,
        "synced_at": synced_at,
        "cached": False,
    }
    await cache_set(cache_key, payload, cfg.cache_ttl)
    log.info(
        "Document graph synced",
        document_id=document_id,
        user_id=user.id,
        concept_count=len(concept_rows),
        neo4j_synced=neo4j_synced,
    )
    return payload


async def get_document_graph(client: Client, document_id: str, user: AuthUser) -> dict[str, Any]:
    doc = get_accessible_document(client, document_id, user, min_role="viewer")
    try:
        concepts = (
            client.table("document_concepts")
            .select("concept_id, name, topic, chunk_indexes")
            .eq("document_id", document_id)
            .order("name")
            .execute()
            .data
            or []
        )
        relationships = (
            client.table("concept_relationships")
            .select("source_concept_id, target_concept_id, relationship_type")
            .eq("document_id", document_id)
            .execute()
            .data
            or []
        )
    except Exception as exc:
        if is_missing_phase2_schema(exc):
            return {
                "document_id": document_id,
                "filename": doc["filename"],
                "neo4j_synced_at": doc.get("neo4j_synced_at"),
                "nodes": [],
                "edges": [],
                "migration_required": True,
                "notice": PHASE2_MIGRATION_NOTICE,
            }
        raise

    return {
        "document_id": document_id,
        "filename": doc["filename"],
        "neo4j_synced_at": doc.get("neo4j_synced_at"),
        "nodes": [
            {
                "id": row["concept_id"],
                "label": row["name"],
                "topic": row.get("topic"),
                "chunk_indexes": row.get("chunk_indexes") or [],
            }
            for row in concepts
        ],
        "edges": [
            {
                "source": row["source_concept_id"],
                "target": row["target_concept_id"],
                "type": row["relationship_type"],
            }
            for row in relationships
        ],
    }
