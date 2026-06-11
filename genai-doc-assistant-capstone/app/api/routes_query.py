from fastapi import APIRouter, Request
from pydantic import BaseModel

from pycorekit.logging.logger import get_logger
from pycorekit.tracing.decorators import with_observability
from pycorekit.tracing.tracing import start_trace
from pycorekit.correlation.context import get_current_correlation_id
from app.core.settings import settings
from app.core.agent_graph import create_graph
from app.service.db_connection import (
    async_load_chat_history,
    async_save_chat_message
)

router = APIRouter(tags=["Query"])
log = get_logger("ask")

graph = create_graph()

# Observability is added automatically by the @with_observability decorator.
# Route handlers should return plain dict responses and let the decorator
# attach the completed sanitized trace after the request finishes.


class QueryRequest(BaseModel):
    question: str
    thread_id: str


@router.post(
    "/ask-question",
    summary="Ask a natural language question using multi‑agent RAG with document selection",
    description=(
        "Runs a LangGraph multi‑agent workflow consisting of:\n"
        "- Planner Agent\n"
        "- Document Selector Agent\n"
        "- Retriever Agent\n"
        "- Reasoning Agent\n"
        "- Response Agent\n\n"
        "Features:\n"
        "- Chat history in ChromaDB\n"
        "- Confidence scoring\n"
        "- Hallucination detection\n"
        "- Guardrails\n"
        "- HITL document selection\n"
    ),
    operation_id="askQuestion"
)
@with_observability("ask_question")
async def ask_question(payload: QueryRequest, request: Request):
    question = payload.question.strip()
    thread_id = payload.thread_id.strip()

    cid = get_current_correlation_id() or "unknown"
    log.info(f"User question: {question} | thread={thread_id}")

    # 1. Load Chat History
    with start_trace("load_chat_history_async", inputs={"thread_id": thread_id}):
        chat_history = await async_load_chat_history(thread_id)

    # 2. Planner + Document Selector
    with start_trace("agent_graph_initial", inputs={"question": question}):
        partial_result = await graph.ainvoke({
            "question": question,
            "chat_history": chat_history
        })

    selected_doc_id = partial_result.get("selected_doc_id")
    needs_user_choice = partial_result.get("needs_user_choice", False)
    candidate_docs = partial_result.get("candidate_docs", [])

    # 3. Ambiguous Document Selection
    if needs_user_choice:
        with start_trace("document_selection_ambiguous"):
            return {
                "thread_id": thread_id,
                "question": question,
                "needs_user_choice": True,
                "candidate_documents": candidate_docs,
                "correlation_id": cid,
            }

    # 4. Full Pipeline Already Ran
    with start_trace("final_answer_pipeline"):
        final_answer = partial_result["final_answer"]
        confidence = partial_result.get("confidence", 0.5)
        hallucinated = partial_result.get("hallucinated", False)

    # 5. Save Chat History
    with start_trace("save_chat_history_async"):
        await async_save_chat_message(thread_id, "user", question)
        await async_save_chat_message(thread_id, "assistant", final_answer)

    # 6. Return Final Answer

    result = {
        "thread_id": thread_id,
        "question": question,
        "answer": final_answer,
        "confidence": confidence,
        "hallucinated": hallucinated,
        "selected_doc_id": selected_doc_id,
        "model": settings.models.llm_model,
        "steps": {
            "planner": partial_result.get("steps"),
            "retrieved_chunks": partial_result.get("retrieved_chunks"),
            "reasoning": partial_result.get("reasoning_summary")
        },
        "correlation_id": cid,
    }
    log.info("Returning response", result=result)
    with start_trace("return_response", inputs={"correlation_id": cid, "thread_id": thread_id}):
        return result

