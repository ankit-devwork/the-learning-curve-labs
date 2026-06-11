from app.core.agent_graph import create_query_graph, create_answer_graph


def test_graphs_compile():
    query_graph = create_query_graph()
    answer_graph = create_answer_graph()
    assert query_graph is not None
    assert answer_graph is not None
