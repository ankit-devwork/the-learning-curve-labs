from app.core.agent_state import AgentState
from app.core.llm import llm
from pycorekit.core_logging.logger import get_logger
from pycorekit.tracing.tracing import start_trace
from pycorekit.observability.observability import observe_llm
from pycorekit.correlation.context import get_current_correlation_id

log = get_logger("planner")


def _format_chat_history(chat_history: list[dict]) -> str:
    if not chat_history:
        return "No prior messages."
    lines = []
    for msg in chat_history[-8:]:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


async def planner_agent(state: AgentState):
    with start_trace("planner_agent", inputs={"question": state.get("question")}):
        log.info("Planner agent started")

        question = state["question"]
        chat_history = state.get("chat_history", [])
        history_text = _format_chat_history(chat_history)

        prompt = (
            "You are the Planner Agent for a document RAG system.\n"
            "Analyze the user question and recent chat history.\n"
            "Return a concise 2-4 step execution plan for answering the question "
            "using document selection, retrieval, reasoning, and response.\n"
            "Do not answer the question itself.\n\n"
            f"Question:\n{question}\n\n"
            f"Chat history:\n{history_text}\n\n"
            "Execution plan:"
        )

        steps = await observe_llm("planner_plan", llm, prompt=prompt)
        log.info("Planner agent finished")

        return {
            **state,
            "steps": steps.strip(),
        }
