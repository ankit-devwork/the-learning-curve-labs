import os
import logging
from typing import List, Dict, Any
from app.core.load_property import settings
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from app.api.routes.document_ingestion import local_vector_engine  # Points to vector_service.py
from app.core.graph_db import graph_service  # Points to graph_service.py

# Initialize a dedicated logger for tracking retrieval operations within this service module
logger = logging.getLogger("app.services.graph_rag_service")

class GraphRAGService:
    """
    Unified Hybrid GraphRAG Engine.
    Blends unstructured dense vector lookups (pgvector) with structured multi-hop 
    graph traversals (Neo4j) to form a robust context matrix for the LLM.
    """

    async def retrieve_vector_only_context(self, query: str, db: AsyncSession, limit: int = 5) -> List[str]:
        """
        Isolated Vector Retrieval Engine Pass.
        Performs an un-hybridized pgvector cosine similarity search sweep.
        Used when the LangGraph orchestration nodes explicitly request isolated vector data.
        
        Args:
            query (str): The natural language query or rephrased search target.
            db (AsyncSession): Active relational database session handle for tracking pgvector queries.
            limit (int): Maximum number of text segments to return. Defaults to 5.
            
        Returns:
            List[str]: A list of raw text strings representing the most similar text chunks.
        """
        logger.info(f"Executing isolated pgvector database segment sweep for query: '{query}'")
        vector_context_chunks = []
        try:
            # 1a. Encode human query into a raw list of embedding weights using the local transformer model
            query_embedding_list = local_vector_engine.encode(query).tolist()
            query_embedding = str(query_embedding_list)
            
            # 1b. Formulate native pgvector cosine distance query match criteria (<=> operator denotes cosine distance)
            vector_query = text("""
                SELECT content 
                FROM document_chunks
                ORDER BY embedding <=> :embedding
                LIMIT :limit;
            """)
            
            # 1c. Execute search query pass against PostgreSQL relational context tables
            vector_results = await db.execute(vector_query, {"embedding": query_embedding, "limit": limit})
            vector_context_chunks = [row[0] for row in vector_results.fetchall()]
            
            logger.info(f"Retrieved {len(vector_context_chunks)} relevant text chunks from pgvector.")
            return vector_context_chunks
            
        except Exception as vec_err:
            # Catch exceptions transparently to prevent pipeline execution crashes during transient database hiccups
            logger.error(f"Isolated vector search pipeline choked temporarily: {vec_err}")
            return []

    async def retrieve_vector_context(self, query: str, db: AsyncSession, limit: int = 5) -> List[str]:
        """
        Alias Framework Mapping Method.
        Proxies requests pointing to alternative graph routing canvas naming conventions 
        to ensure runtime orchestration continuity.
        """
        return await self.retrieve_vector_only_context(query=query, db=db, limit=limit)

    async def retrieve_hybrid_context(self, query: str, db: AsyncSession, entities: List[str] = None, limit: int = 5) -> str:
        """
        Executes a dual-database parallel-safe context retrieval pass.
        Combines semantic relational blocks with structured entity graphs into a unified ledger.
        
        Args:
            query (str): The raw text prompt submitted by the user.
            db (AsyncSession): Active transactional database session context.
            entities (List[str]): Extracted target keywords from the upstream Router Node.
            limit (int): Number of relational context blocks to gather.
            
        Returns:
            str: A combined context ledger structured cleanly for the LLM Synthesis Node.
        """
        logger.info(f"Executing hybrid GraphRAG retrieval pass for query: '{query}'")
        
        # -----------------------------------------------------------------------------------------
        # PHASE 1: DENSE UNSTRUCTURED VECTOR RETRIEVAL (pgvector)
        # -----------------------------------------------------------------------------------------
        # Reuses the isolated internal method to gather unstructured semantic text segments
        vector_context_chunks = await self.retrieve_vector_only_context(query=query, db=db, limit=limit)

        # -----------------------------------------------------------------------------------------
        # PHASE 2: UNIVERSAL STRUCTURED GRAPH ENTITY TRAVERSAL (Neo4j)
        # -----------------------------------------------------------------------------------------
        # Crawls relationship neighborhoods around entities extracted dynamically by the Router Node.
        # This implementation contains ZERO domain-specific hardcoding.
        graph_context_elements = []
        
        # Generic multi-hop matching query targeting standardized graph schemas (Agent, Risk, and Task)
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
        
        # Guard clause: Trust the Router Node explicitly. 
        # If no semantic entities were isolated, bypass graph table sweeps to preserve CPU cycles.
        if entities:
            try:
                async with driver.session() as session:
                    # Iterate transparently through whatever terms the routing canvas extracted
                    for term in entities:
                        logger.info(f"[Neo4j Cypher Scan] Dynamically crawling graph for entity: '{term}'")
                        
                        # Execute async cypher query passing the entity token
                        result = await session.run(graph_query, parameters={"search_term": term})
                        
                        # Consume the async stream using list comprehension to handle driver streaming variants
                        records = [record async for record in result]
                        
                        # Process records into human-readable context sentences based on active label variants
                        for rec in records:
                            data = rec.data()
                            node_type = data["node_type"]        
                            node_props = data["node_props"]      
                            target_type = data["target_type"]    
                            target_props = data["target_props"]  
                            rel = data["rel_type"]               
                            
                            # Construct semantic strings universally based on properties, not raw values
                            if node_type == "Agent":
                                agent_name = node_props.get('name', 'Unnamed Entity')
                                target_action = target_props.get('action') or target_props.get('filename', 'Unknown Reference')
                                line = f"Agent ({agent_name}) is verified as [{rel}] connected to -> {target_type} ({target_action})."
                                
                            elif node_type == "Risk":
                                category = node_props.get('category', 'General Unclassified')
                                severity = node_props.get('severity', 'Unrated Exposure')
                                line = f"Risk Parameter [{category} - Severity Level: {severity}] is actively linked to structural element {target_type}."
                                
                            else:  # Node Type is structural Task
                                # Check 'task' first to match your DocumentInsightsSchema extraction property!
                                action = node_props.get('task') or node_props.get('action') or 'Unnamed Action Item'
                                assignee = node_props.get('assignee', 'Unassigned')
                                raw_deadline = node_props.get('deadline')
                                
                                # Format milestones gracefully depending on if data boundaries exist
                                if not raw_deadline or str(raw_deadline).strip().lower() in ['none', 'null', '']:
                                    deadline_phrase = "no explicit milestone timeline is specified inside the current source material"
                                else:
                                    deadline_phrase = f"a scheduled deadline tracking flag of {raw_deadline}"
                                    
                                line = f"System Task Asset ({action}) assigned to ({assignee}) is registered on schedule with {deadline_phrase}."
                                
                            graph_context_elements.append(line)
                            
            except Exception as graph_err:
                # Catch internal graph traversal exceptions safely to ensure system availability
                logger.error(f"Graph structural traversal stuttered safely: {graph_err}")
                pass
        else:
            logger.info("[Neo4j Cypher Scan] Skipped. Router provided 0 entities for structured lookup.")

        # De-duplicate entries to prevent token padding in the prompt matrix
        unique_graph_statements = list(set(graph_context_elements))
        logger.info(f"Retrieved {len(unique_graph_statements)} unique structural context statements from Neo4j.")

        # -----------------------------------------------------------------------------------------
        # PHASE 3: CONTEXT MATRIX COMPOSITION
        # -----------------------------------------------------------------------------------------
        # Compile raw segments into a strongly delimited document bundle for the Synthesis Node
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

        # Return a neatly assembled textual ledger matching instructions for the synthesis generation loop
        return "\n".join(context_matrix)

# Instantiate singleton instance for consumption across the agent graph routing topology
graph_rag_service = GraphRAGService()
