from app.core.graph_db import graph_service
from app.observability.logger import logger

async def sync_document_to_knowledge_graph(document_id: str, filename: str, insights: dict):
    """
    Parses structural extraction insights from Groq models and builds
    interconnected conceptual graph node hierarchies inside Neo4j.
    """
    logger.info(f"[{document_id}] Kicking off knowledge graph synchronization phase...")

    driver = graph_service.get_driver()

    try:
        async with driver.session() as session:
            
            # 1. Merge parent Document tracking root node
            doc_query = """
            MERGE (d:Document {id: $doc_id})
            SET d.filename = $filename, 
                d.updated_at = timestamp()
            RETURN d
            """
            # 🚀 FIX: Pass parameters explicitly as an isolated dict argument
            await session.run(doc_query, parameters={"doc_id": document_id, "filename": filename})

            # 2. Iterate and stitch extracted liability risks
            risks = insights.get("risks", [])
            for idx, risk_node in enumerate(risks):
                risk_query = """
                MATCH (d:Document {id: $doc_id})
                MERGE (r:Risk {id: $risk_uid})
                SET r.category = $category,
                    r.description = $description,
                    r.severity = $severity
                MERGE (d)-[:POSES_RISK]->(r)
                """
                await session.run(risk_query, parameters={
                    "doc_id": document_id,
                    "risk_uid": f"{document_id}_risk_{idx}",
                    "category": risk_node.get("category", "General Liability"),
                    "description": risk_node.get("description", ""),
                    "severity": risk_node.get("severity", "Medium")
                })

            # 3. Iterate and stitch transactional tasks + assignees
            tasks = insights.get("tasks_and_deadlines", [])
            for idx, task_node in enumerate(tasks):
                assignee_name = task_node.get("assignee", "Unassigned").strip()
                if not assignee_name:
                    assignee_name = "Unassigned"

                task_query = """
                MATCH (d:Document {id: $doc_id})
                
                // Generate or capture unique owner node properties
                MERGE (a:Agent {name: $assignee})
                
                // Connect structural task targets
                MERGE (t:Task {id: $task_uid})
                SET t.action = $action,
                    t.deadline = $deadline
                    
                // Establish systemic context linking paths
                MERGE (d)-[:CONTAINS_TASK]->(t)
                MERGE (a)-[:ASSIGNED_TO]->(t)
                """
                await session.run(task_query, parameters={
                    "doc_id": document_id,
                    "task_uid": f"{document_id}_task_{idx}",
                    "assignee": assignee_name,
                    "action": task_node.get("task", ""),
                    "deadline": task_node.get("deadline", "None")
                })

            logger.info(f"[{document_id}] Neo4j graph entities successfully compiled and committed.")

    except Exception as sync_fault:
        logger.error(f"[{document_id}] Graph Synchronizer step critically failed: {sync_fault}")
        pass