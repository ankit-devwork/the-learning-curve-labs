from langgraph.graph import StateGraph, END
from app.core.agent_state import AgentState

from app.agents.planner import planner_agent
from app.agents.retriever import retriever_agent
from app.agents.reasoning import reasoning_agent
from app.agents.response import response_agent
from app.agents.document_selector import document_selector_agent


def _route_after_selector(state: AgentState) -> str:
    """Pause the pipeline when the user must pick a document (HITL)."""
    if state.get("needs_user_choice"):
        return "hitl_end"
    return "retriever"


def create_query_graph():
    """
    Full query graph with HITL pause after document selection.

    Flow:
      planner -> document_selector -> (retriever -> reasoning -> response | END)
    """
    graph = StateGraph(AgentState)

    graph.add_node("planner", planner_agent)
    graph.add_node("document_selector", document_selector_agent)
    graph.add_node("retriever", retriever_agent)
    graph.add_node("reasoning", reasoning_agent)
    graph.add_node("response", response_agent)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "document_selector")
    graph.add_conditional_edges(
        "document_selector",
        _route_after_selector,
        {
            "hitl_end": END,
            "retriever": "retriever",
        },
    )
    graph.add_edge("retriever", "reasoning")
    graph.add_edge("reasoning", "response")
    graph.add_edge("response", END)

    return graph.compile()


def create_answer_graph():
    """
    Resume graph used after HITL document selection.

    Flow:
      retriever -> reasoning -> response
    """
    graph = StateGraph(AgentState)

    graph.add_node("retriever", retriever_agent)
    graph.add_node("reasoning", reasoning_agent)
    graph.add_node("response", response_agent)

    graph.set_entry_point("retriever")
    graph.add_edge("retriever", "reasoning")
    graph.add_edge("reasoning", "response")
    graph.add_edge("response", END)

    return graph.compile()
