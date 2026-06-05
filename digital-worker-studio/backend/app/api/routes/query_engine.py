from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from langchain_core.messages import HumanMessage, AIMessage

from app.core.database import db_service
import app.agents.agent_graph as agent_module
from app.services.cache_service import cache_service
from app.observability.logger import get_request_logger
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

router = APIRouter(prefix="/api/worker/query", tags=["GraphRAG Search Engine"])


class QueryRequestSchema(BaseModel):
    question: str
    thread_id: str


@router.post("/ask")
async def execute_graph_rag_query(
    request: Request,
    payload: QueryRequestSchema,
    db: AsyncSession = Depends(db_service.get_session),
):
    question_text = payload.question.strip()
    thread_session_id = payload.thread_id.strip()

    log = get_request_logger(request, thread_id=thread_session_id)

    # ---------------------------------------------------------
    # FINAL ANSWER CACHE CHECK (NEW)
    # ---------------------------------------------------------
    cached_answer = await cache_service.get_final_answer(
        thread_session_id, question_text
    )

    if cached_answer:
        log.info(f"[FINAL ANSWER CACHE HIT] thread={thread_session_id}")
        return {
            "status": "success",
            "query": question_text,
            "answer": cached_answer["answer"],
        }

    log.info(f"[FINAL ANSWER CACHE MISS] thread={thread_session_id}")

    # ---------------------------------------------------------
    # Ensure LangGraph executor is initialized
    # ---------------------------------------------------------
    if not hasattr(agent_module, "graph_rag_executor") or agent_module.graph_rag_executor is None:
        log.warning("[Lazy Init] Compiling LangGraph executor...")
        if hasattr(agent_module, "pool") and agent_module.pool is not None:
            postgres_checkpointer = AsyncPostgresSaver(agent_module.pool)
            agent_module.graph_rag_executor = agent_module.workflow.compile(
                checkpointer=postgres_checkpointer
            )
        else:
            raise HTTPException(
                status_code=503,
                detail="Checkpoint pool is uninitialized.",
            )

    # ---------------------------------------------------------
    # RUN FULL GRAPHRAG PIPELINE
    # ---------------------------------------------------------
    initial_state = {
        "query": question_text,
        "retrieved_context": "",
        "routing_decision": "",
        "final_answer": "",
    }

    log.info(f"[API Gateway] Executing LangGraph for thread={thread_session_id}...")
    final_output_state = await agent_module.graph_rag_executor.ainvoke(
        initial_state,
        config={"configurable": {"thread_id": thread_session_id}},
    )

    final_answer_string = final_output_state.get("final_answer", "").strip()
    if not final_answer_string:
        raise HTTPException(
            status_code=500,
            detail="LangGraph execution completed but no final answer was produced.",
        )

    # ---------------------------------------------------------
    # STORE FINAL ANSWER IN CACHE (NEW)
    # ---------------------------------------------------------
    await cache_service.set_final_answer(
        thread_session_id,
        question_text,
        {"answer": final_answer_string},
    )

    log.info("[API Gateway] Returning final answer.")
    return {
        "status": "success",
        "query": question_text,
        "answer": final_answer_string,
    }
