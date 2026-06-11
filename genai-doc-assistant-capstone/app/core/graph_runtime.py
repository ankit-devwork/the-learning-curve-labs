"""
Shared compiled LangGraph instances.

Compiled once at import time and reused across API routes.
"""

from app.core.agent_graph import create_query_graph, create_answer_graph

query_graph = create_query_graph()
answer_graph = create_answer_graph()
