from pycorekit.tracing.tracing import start_trace
from pycorekit.observability.observability import observe_llm
from pycorekit.correlation.context import get_current_correlation_id
from app.core.agent_state import AgentState
from pycorekit.core_logging.logger import get_logger
from app.core.llm import llm
log = get_logger("reasoning")


async def reasoning_agent(state: AgentState):
    cid = get_current_correlation_id() or "unknown"

    with start_trace("reasoning_agent", inputs={"state": dict(state)}):
        log.info("Reasoning agent started")

        question = state["question"]
        chat_history = state["chat_history"]
        chunks = state.get("retrieved_chunks", [])

        # 1. Build context
        with start_trace("build_context", inputs={"num_chunks": len(chunks)}):
            safe_chunks = [str(c) for c in chunks]
            context = "\n\n".join(safe_chunks)

        # 2. Build reasoning prompt
        reasoning_prompt = (
            "You are the Reasoning Agent.\n"
            "Your job is to analyze the question, chat history, and retrieved context.\n"
            "Produce a structured reasoning summary that will help the Response Agent.\n\n"
            f"Question:\n{question}\n\n"
            f"Chat history:\n{chat_history}\n\n"
            f"Context:\n{context}\n\n"
            "Now produce a reasoning summary:"
        )

        # 3. Generate reasoning summary (LLM call)
        reasoning = await observe_llm("generate_reasoning_summary", llm, prompt=reasoning_prompt)

        log.info("Reasoning agent finished")

        return {
            **state,
            "reasoning_summary": reasoning
        }
