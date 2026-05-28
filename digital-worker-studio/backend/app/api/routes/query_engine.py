from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from langchain_core.messages import HumanMessage, AIMessage

# Project Aligned Architecture and Caching Engine Frameworks
from app.core.database import db_service
import app.agents.agent_graph as agent_module  
from app.services.cache_service import cache_service  
from app.observability.logger import logger
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
router = APIRouter(prefix="/api/worker/query", tags=["GraphRAG Search Engine"])


class QueryRequestSchema(BaseModel):
    """Defines the data contract validation schema for natural language lookups with chat history."""
    question: str = Field(
        ..., 
        description="The natural language question you want to ask the cognitive document graph engine.",
        example="What critical tasks are assigned to MetLife and what are their deadlines?"
    )
    thread_id: str = Field(
        ...,
        description="Unique string or UUID session identifier used by the checkpointer to load and save chat logs.",
        example="chat-session-xyz-123"
    )


@router.post("/ask")
async def execute_graph_rag_query(
    payload: QueryRequestSchema,
    db: AsyncSession = Depends(db_service.get_session)
):
    """
    Processes a natural language question through a high-performance cognitive search loop.
    """
    question_text = payload.question.strip()
    thread_session_id = payload.thread_id.strip()
    
    cache_lookup_key = f"query:{thread_session_id}:{question_text.lower()}"
    session_config = {"configurable": {"thread_id": thread_session_id}}
    
    try:
        # 🚀 CACHE HIT PROCESSING LAYER
        # Inside app/api/routes/query_engine.py -> execute_graph_rag_query function

        # 🚀 CACHE HIT PROCESSING LAYER
        cached_payload = await cache_service.get_with_sliding_ttl(cache_lookup_key)
        if cached_payload:
            logger.info(f"[API Gateway Cache Hit] Short-circuiting execution for thread: {thread_session_id}")
            
            if not hasattr(agent_module, "graph_rag_executor") or agent_module.graph_rag_executor is None:
                if hasattr(agent_module, "pool") and agent_module.pool is not None:
                   
                    postgres_checkpointer = AsyncPostgresSaver(agent_module.pool)
                    agent_module.graph_rag_executor = agent_module.workflow.compile(checkpointer=postgres_checkpointer)
            
            # Sync the cached turn back to PostgreSQL memory if executor context is ready
            if agent_module.graph_rag_executor is not None:
                try:
                    # 🚀 FIX: Changed 'auupdate_state' to 'aupdate_state'
                    await agent_module.graph_rag_executor.aupdate_state(
                        session_config,
                        {
                            "messages": [
                                HumanMessage(content=question_text),
                                AIMessage(content=cached_payload["answer"])
                            ]
                        },
                        as_node="synthesis"  
                    )
                    logger.info(f"[API Gateway Checkpointer Sync] Synced cached history turn to Postgres thread.")
                except Exception as sync_err:
                    logger.error(f"[API Gateway Checkpointer Warning] Failed to update history on cache hit: {sync_err}")

            return {
                "status": "success",
                "query": question_text,
                "answer": cached_payload["answer"]
            }
            
    except Exception as cache_read_err:
        logger.error(f"[API Gateway Cache Warning] Bypassing Redis read exception: {cache_read_err}")

    logger.info(f"Received secure API search request for thread '{thread_session_id}' | question: '{question_text}'")
    
    try:
        # Lazy Engine compilation fallback safety gate to resolve application start race conditions
        if not hasattr(agent_module, "graph_rag_executor") or agent_module.graph_rag_executor is None:
            logger.warning("[API Gateway Lazy Init] Graph executor was uninitialized at call invocation. Compiling canvas fallback context...")
            if hasattr(agent_module, "pool") and agent_module.pool is not None:
                from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
                postgres_checkpointer = AsyncPostgresSaver(agent_module.pool)
                agent_module.graph_rag_executor = agent_module.workflow.compile(checkpointer=postgres_checkpointer)
            else:
                raise HTTPException(
                    status_code=503,
                    detail="Relational database storage resource checkpoint pool handles are currently uninitialized."
                )

        # 🚀 FIX: Completely removed "db_session": db from the initial_state dict.
        # This keeps the dictionary fields completely primitive (strings) so the initial checkpoint save passes msgpack validation.
        initial_state = {
            "query": question_text,
            "retrieved_context": "",       
            "routing_decision": "",        
            "final_answer": ""             
        }
        
        logger.info(f"[API Gateway] Submitting query state to LangGraph executor canvas for thread: {thread_session_id}...")
        final_output_state = await agent_module.graph_rag_executor.ainvoke(initial_state, config=session_config)
        
        # 3. Validate and Extract Response Assertions
        final_answer_string = final_output_state.get("final_answer", "").strip()
        if not final_answer_string or final_answer_string == "Failed to compile execution response string.":
            raise ValueError("LangGraph execution loop completed but failed to generate a grounded answer string.")

        # 4. Structure payload for Redis caching engine 
        cache_payload = {
            "answer": final_answer_string
        }
        
        try:
            await cache_service.set_fixed_ttl(cache_lookup_key, cache_payload)
        except Exception as cache_write_err:
            logger.error(f"[API Gateway Cache Warning] Failed to commit new entry payload to Redis grid: {cache_write_err}")

        logger.info("[API Gateway] LangGraph loop finished execution successfully. Returning structured response.")
        return {
            "status": "success",
            "query": question_text,
            "answer": final_answer_string
        }
        
    except HTTPException:
        raise
    except Exception as graph_runtime_err:
        logger.error(f"[API Gateway Critical Fault] LangGraph execution workflow loop collapsed: {graph_runtime_err}")
        raise HTTPException(
            status_code=500, 
            detail=f"GraphRAG Cognitive Engine runtime execution fault: {str(graph_runtime_err)}"
        )