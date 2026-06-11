from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel

from pycorekit.core_logging.logger import get_logger
from pycorekit.tracing.decorators import with_observability
from pycorekit.tracing.tracing import start_trace
from pycorekit.correlation.context import get_current_correlation_id
from app.core.graph_runtime import answer_graph
from app.service.db_connection import (
    async_load_chat_history,
    async_save_chat_message,
)
from app.service.query_cache import get_cached_answer, set_cached_answer

router = APIRouter(tags=["Query"])
log = get_logger("choose_document")


class ChooseDocumentRequest(BaseModel):
    thread_id: str
    question: str
    selected_doc_id: str


@router.post(
    "/choose-document",
    summary="Resume the RAG pipeline after user selects a document",
    operation_id="chooseDocument",
)
@with_observability("choose_document")
async def choose_document(payload: ChooseDocumentRequest, request: Request):
    thread_id = payload.thread_id.strip()
    question = payload.question.strip()
    selected_doc_id = payload.selected_doc_id.strip()
    cid = get_current_correlation_id() or "unknown"

    cache_key_suffix = f"{selected_doc_id}"
    cache_question = f"{question}::doc={cache_key_suffix}"
    cached = await get_cached_answer(thread_id, cache_question)
    if cached:
        cached["cache_hit"] = True
        cached["correlation_id"] = cid
        return cached

    log.info(
        f"Resuming pipeline with selected_doc_id={selected_doc_id} | thread={thread_id}"
    )

    with start_trace("load_chat_history_async", inputs={"thread_id": thread_id}):
        chat_history = await async_load_chat_history(thread_id)

    with start_trace("answer_graph_resume", inputs={"selected_doc_id": selected_doc_id}):
        result = await answer_graph.ainvoke({
            "question": question,
            "chat_history": chat_history,
            "selected_doc_id": selected_doc_id,
            "needs_user_choice": False,
        })

    final_answer = result.get("final_answer", "").strip()
    if not final_answer:
        raise HTTPException(status_code=500, detail="Resume pipeline completed without a final answer.")

    confidence = result.get("confidence", 0.5)
    hallucinated = result.get("hallucinated", False)

    with start_trace("save_chat_history_async"):
        await async_save_chat_message(thread_id, "user", question)
        await async_save_chat_message(thread_id, "assistant", final_answer)

    response = {
        "thread_id": thread_id,
        "question": question,
        "selected_doc_id": selected_doc_id,
        "answer": final_answer,
        "confidence": confidence,
        "hallucinated": hallucinated,
        "cache_hit": False,
        "steps": {
            "retrieved_chunks": result.get("retrieved_chunks"),
            "reasoning": result.get("reasoning_summary"),
        },
        "correlation_id": cid,
    }

    await set_cached_answer(thread_id, cache_question, response)
    return response
