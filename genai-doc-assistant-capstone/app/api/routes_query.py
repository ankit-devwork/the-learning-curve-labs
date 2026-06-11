from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel

from pycorekit.core_logging.logger import get_logger
from pycorekit.tracing.decorators import with_observability
from pycorekit.tracing.tracing import start_trace
from pycorekit.correlation.context import get_current_correlation_id
from app.core.settings import settings
from app.core.graph_runtime import query_graph
from app.service.db_connection import (
    async_load_chat_history,
    async_save_chat_message,
)
from app.service.query_cache import get_cached_answer, set_cached_answer

router = APIRouter(tags=["Query"])
log = get_logger("ask")


class QueryRequest(BaseModel):
    question: str
    thread_id: str


@router.post(
    "/ask-question",
    summary="Ask a natural language question using multi‑agent RAG with document selection",
    operation_id="askQuestion",
)
@with_observability("ask_question")
async def ask_question(payload: QueryRequest, request: Request):
    question = payload.question.strip()
    thread_id = payload.thread_id.strip()
    cid = get_current_correlation_id() or "unknown"

    log.info(f"User question: {question} | thread={thread_id}")

    cached = await get_cached_answer(thread_id, question)
    if cached:
        log.info("Returning cached answer", thread_id=thread_id)
        cached["cache_hit"] = True
        cached["correlation_id"] = cid
        return cached

    with start_trace("load_chat_history_async", inputs={"thread_id": thread_id}):
        chat_history = await async_load_chat_history(thread_id)

    with start_trace("agent_graph_query", inputs={"question": question}):
        partial_result = await query_graph.ainvoke({
            "question": question,
            "chat_history": chat_history,
        })

    if partial_result.get("needs_user_choice"):
        return {
            "thread_id": thread_id,
            "question": question,
            "needs_user_choice": True,
            "candidate_documents": partial_result.get("candidate_docs", []),
            "correlation_id": cid,
            "cache_hit": False,
        }

    final_answer = partial_result.get("final_answer", "").strip()
    if not final_answer:
        raise HTTPException(status_code=500, detail="Pipeline completed without a final answer.")

    confidence = partial_result.get("confidence", 0.5)
    hallucinated = partial_result.get("hallucinated", False)
    selected_doc_id = partial_result.get("selected_doc_id")

    with start_trace("save_chat_history_async"):
        await async_save_chat_message(thread_id, "user", question)
        await async_save_chat_message(thread_id, "assistant", final_answer)

    result = {
        "thread_id": thread_id,
        "question": question,
        "answer": final_answer,
        "confidence": confidence,
        "hallucinated": hallucinated,
        "selected_doc_id": selected_doc_id,
        "model": settings.models.llm_model,
        "cache_hit": False,
        "steps": {
            "planner": partial_result.get("steps"),
            "retrieved_chunks": partial_result.get("retrieved_chunks"),
            "reasoning": partial_result.get("reasoning_summary"),
        },
        "correlation_id": cid,
    }

    await set_cached_answer(thread_id, question, result)
    log.info("Returning response", thread_id=thread_id)
    return result
