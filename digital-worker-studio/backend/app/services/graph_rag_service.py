import os
import logging
from typing import List, Dict, Any
from app.core.load_property import settings
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from app.api.routes.document_ingestion import local_vector_engine
from app.core.graph_db import graph_service

logger = logging.getLogger("app.services.graph_rag_service")


# -------------------------------------------------------------------
# NEW: Required pgvector literal converter
# -------------------------------------------------------------------
def to_pgvector_literal(vec: List[float]) -> str:
    """
    Converts a Python list of floats into a pgvector-compatible literal string.
    Example: [0.1, -0.2, 0.3] → "[0.1, -0.2, 0.3]"
    """
    return "[" + ", ".join(f"{float(x):.6f}" for x in vec) + "]"


class GraphRAGService:
    """
    Unified Hybrid GraphRAG Engine.
    Blends unstructured dense vector lookups (pgvector) with structured multi-hop 
    graph traversals (Neo4j) to form a robust context matrix for the LLM.
    """

    # -------------------------------------------------------------------
    # VECTOR-ONLY RETRIEVAL (PATCHED)
    # -------------------------------------------------------------------
    async def retrieve_vector_only_context(self, query: str, db: AsyncSession, limit: int = 5) -> List[str]:
        logger.info(f"Executing isolated pgvector database segment sweep for query: '{query}'")
        vector_context_chunks = []

        try:
            # Encode query → embedding list
            try:
                query_embedding_list = local_vector_engine.encode(query).tolist()
            except Exception:
                import asyncio
                query_embedding_list = await asyncio.to_thread(
                    lambda: local_vector_engine.encode(query).tolist()
                )

            # Convert Python list → pgvector literal
            pg_embedding = to_pgvector_literal(query_embedding_list)

            vector_query = text("""
                SELECT content 
                FROM document_chunks
                ORDER BY embedding <=> :embedding
                LIMIT :limit;
            """)

            result = await db.execute(
                vector_query,
                {"embedding": pg_embedding, "limit": limit}
            )

            rows = result.mappings().all()
            vector_context_chunks = [row.get("content") for row in rows]

            logger.info(f"Retrieved {len(vector_context_chunks)} relevant text chunks from pgvector.")
            return vector_context_chunks

        except Exception as vec_err:
            logger.error(f"Isolated vector search pipeline choked temporarily: {vec_err}")
            return []

    # Alias
    async def retrieve_vector_context(self, query: str, db: AsyncSession, limit: int = 5) -> List[str]:
        return await self.retrieve_vector_only_context(query=query, db=db, limit=limit)

    # -------------------------------------------------------------------
    # HYBRID GRAPH + VECTOR RETRIEVAL (UNCHANGED)
    # -------------------------------------------------------------------
    async def retrieve_hybrid_context(self, query: str, db: AsyncSession, entities: List[str] = None, limit: int = 5) -> str:
        logger.info(f"Executing hybrid GraphRAG retrieval pass for query: '{query}'")

        # PHASE 1 — VECTOR
        vector_context_chunks = await self.retrieve_vector_only_context(query=query, db=db, limit=limit)

        # PHASE 2 — GRAPH
        graph_context_elements = []

        graph_query = """
        MATCH (n)
        WHERE (n:Agent OR n:Risk OR n:Task)
          AND (toLower(coalesce(n.name, '')) CONTAINS toLower($search_term) 
               OR toLower(coalesce(n.category, '')) CONTAINS toLower($search_term) 
               OR toLower(coalesce(n.action, '')) CONTAINS toLower($search_term))
        
        MATCH (n)-[r]-(m)
        RETURN labels(n)[0] AS node_type, properties(n) AS node_props, 
               type(r) AS rel_type, labels(m)[0] AS target_type, properties(m) AS target_props
        LIMIT 20
        """

        driver = graph_service.get_driver()

        if entities:
            try:
                async with driver.session() as session:
                    for term in entities:
                        logger.info(f"[Neo4j Cypher Scan] Dynamically crawling graph for entity: '{term}'")

                        result = await session.run(graph_query, parameters={"search_term": term})
                        records = [record async for record in result]

                        for rec in records:
                            data = rec.data()
                            node_type = data["node_type"]
                            node_props = data["node_props"]
                            target_type = data["target_type"]
                            target_props = data["target_props"]
                            rel = data["rel_type"]

                            if node_type == "Agent":
                                agent_name = node_props.get('name', 'Unnamed Entity')
                                target_action = target_props.get('action') or target_props.get('filename', 'Unknown Reference')
                                line = f"Agent ({agent_name}) is verified as [{rel}] connected to -> {target_type} ({target_action})."

                            elif node_type == "Risk":
                                category = node_props.get('category', 'General Unclassified')
                                severity = node_props.get('severity', 'Unrated Exposure')
                                line = f"Risk Parameter [{category} - Severity Level: {severity}] is actively linked to structural element {target_type}."

                            else:
                                action = node_props.get('task') or node_props.get('action') or 'Unnamed Action Item'
                                assignee = node_props.get('assignee', 'Unassigned')
                                raw_deadline = node_props.get('deadline')

                                if not raw_deadline or str(raw_deadline).strip().lower() in ['none', 'null', '']:
                                    deadline_phrase = "no explicit milestone timeline is specified inside the current source material"
                                else:
                                    deadline_phrase = f"a scheduled deadline tracking flag of {raw_deadline}"

                                line = f"System Task Asset ({action}) assigned to ({assignee}) is registered on schedule with {deadline_phrase}."

                            graph_context_elements.append(line)

            except Exception as graph_err:
                logger.error(f"Graph structural traversal stuttered safely: {graph_err}")
        else:
            logger.info("[Neo4j Cypher Scan] Skipped. Router provided 0 entities for structured lookup.")

        unique_graph_statements = list(set(graph_context_elements))
        logger.info(f"Retrieved {len(unique_graph_statements)} unique structural context statements from Neo4j.")

        # PHASE 3 — COMPOSITION
        context_matrix = []

        context_matrix.append("=== UNSTRUCTURED SEMANTIC TEXT CHUNKS (VECTOR SPACE) ===")
        if vector_context_chunks:
            for i, chunk in enumerate(vector_context_chunks, 1):
                context_matrix.append(f"[{i}] {chunk.strip()}")
        else:
            context_matrix.append("No semantic textual snippets retrieved for this query footprint.")

        context_matrix.append("\n=== STRUCTURED KNOWLEDGE CONNECTIVITY (GRAPH SPACE) ===")
        if unique_graph_statements:
            for i, statement in enumerate(unique_graph_statements, 1):
                context_matrix.append(f"({i}) {statement}")
        else:
            context_matrix.append("No explicit structural node/edge connections were extracted matching the identified query entities.")

        return "\n".join(context_matrix)


# Singleton instance
graph_rag_service = GraphRAGService()