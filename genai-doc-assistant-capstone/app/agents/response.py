from app.core.agent_state import AgentState
from app.core.llm import llm

from app.core.guardrails import apply_guardrails
from app.core.guardrails import detect_hallucination_async

from pycorekit.logging.logger import get_logger
log = get_logger("response")

# ✅ Use unified tracing engine
from pycorekit.tracing.tracing import start_trace

# ✅ Use LLM observability wrapper for richer metadata
from pycorekit.observability.observability import observe_llm

from pycorekit.correlation.context import get_current_correlation_id


async def response_agent(state: AgentState):
    cid = get_current_correlation_id() or "unknown"

    # Top-level span for the entire agent
    with start_trace("response_agent", inputs={"state": dict(state)}):
        log.info("Response agent started")

        reasoning = state["reasoning_summary"]
        question = state["question"]
        context_chunks = state.get("retrieved_chunks", [])

        # Build prompt
        prompt = (
            "You are the Response Agent.\n\n"
            "Your job:\n"
            "- Read the reasoning summary\n"
            "- Produce a clear, concise, grounded answer\n"
            "- Do NOT invent facts not present in the retrieved context\n"
            "- Do NOT include system messages\n"
            "- Answer the user's question directly\n\n"
            f"User question:\n{question}\n\n"
            f"Reasoning summary:\n{reasoning}\n\n"
            "Now provide the final answer:\n"
        )

        # 1. Generate answer (LLM call with full observability)
        answer = await observe_llm("generate_answer", llm, prompt=prompt)

        # 2. Apply guardrails
        with start_trace("apply_guardrails"):
            answer = apply_guardrails(answer)

        # 3. Hallucination detection
        with start_trace("hallucination_detection"):
            context = "\n\n".join(context_chunks)
            hallucinated = await detect_hallucination_async(answer, context)
            confidence = 1.0 if not hallucinated else 0.4

        log.info(
            f"Response agent finished | hallucinated={hallucinated} | confidence={confidence}"
        )

        return {
            **state,
            "final_answer": answer,
            "confidence": confidence,
            "hallucinated": hallucinated
        }
