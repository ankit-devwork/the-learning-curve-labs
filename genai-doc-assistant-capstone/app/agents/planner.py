from app.core.agent_state import AgentState
from pycorekit.logging.logger import get_logger

# ✅ Use unified tracing engine
from pycorekit.tracing.tracing import start_trace

from pycorekit.correlation.context import get_current_correlation_id

log = get_logger("planner")


async def planner_agent(state: AgentState):
    cid = get_current_correlation_id() or "unknown"

    # Top-level span for the planner agent
    with start_trace("planner_agent", inputs={"state": dict(state)}):
        log.info("Planner agent started")

        question = state["question"]
        chat_history = state["chat_history"]

        # The planner defines the pipeline steps
        steps = (
            "document_selection → retrieval → reasoning → response"
        )

        log.info("Planner agent finished")

        return {
            **state,
            "steps": steps
        }
