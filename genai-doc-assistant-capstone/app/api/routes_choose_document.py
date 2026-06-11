from fastapi import APIRouter, Request
from pydantic import BaseModel

from pycorekit.logging.logger import get_logger
from pycorekit.tracing.decorators import with_observability
from pycorekit.tracing.tracing import start_trace
from pycorekit.correlation.context import get_current_correlation_id

from app.core.agent_graph import create_graph
from app.service.db_connection import (
    async_load_chat_history,
    async_save_chat_message,
)

router = APIRouter(tags=["Query"])
log = get_logger("choose_document")

graph = create_graph()

# Observability metadata is injected by @with_observability.
# Keep route returns focused on business payload only.


class ChooseDocumentRequest(BaseModel):
    thread_id: str
    question: str
    selected_doc_id: str


@router.post(
    "/choose-document",
    summary="Resume the RAG pipeline after user selects a document",
    operation_id="chooseDocument"
)
@with_observability("choose_document")
async def choose_document(payload: ChooseDocumentRequest, request: Request):

    thread_id = payload.thread_id.strip()
    question = payload.question.strip()
    selected_doc_id = payload.selected_doc_id.strip()

    cid = get_current_correlation_id() or "unknown"

    log.info(
        f"Resuming pipeline with selected_doc_id={selected_doc_id} | thread={thread_id}"
    )

    # 1. Load chat history
    with start_trace("load_chat_history_async", inputs={"thread_id": thread_id}):
        chat_history = await async_load_chat_history(thread_id)

    # 2. Resume LangGraph pipeline
    with start_trace("resume_pipeline", inputs={"selected_doc_id": selected_doc_id}):
        result = await graph.ainvoke({
            "question": question,
            "chat_history": chat_history,
            "selected_doc_id": selected_doc_id
        })

    # 3. Extract final answer
    with start_trace("extract_final_answer"):
        final_answer = result["final_answer"]
        confidence = result.get("confidence", 0.5)
        hallucinated = result.get("hallucinated", False)

    # 4. Save chat history
    with start_trace("save_chat_history_async"):
        await async_save_chat_message(thread_id, "user", question)
        await async_save_chat_message(thread_id, "assistant", final_answer)

    # 5. Format response
    with start_trace("format_response"):
        return {
            "thread_id": thread_id,
            "question": question,
            "selected_doc_id": selected_doc_id,
            "answer": final_answer,
            "confidence": confidence,
            "hallucinated": hallucinated,
            "steps": {
                "retrieved_chunks": result.get("retrieved_chunks"),
                "reasoning": result.get("reasoning_summary")
            },
            "correlation_id": cid,
        }
