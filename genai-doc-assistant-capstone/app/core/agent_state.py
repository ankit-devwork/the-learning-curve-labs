from typing import TypedDict, List, Dict, Optional


class AgentState(TypedDict, total=False):
    # User input
    question: str
    chat_history: List[Dict[str, str]]

    # Document selection
    selected_doc_id: Optional[str]
    needs_user_choice: Optional[bool]
    candidate_docs: Optional[List[Dict]]

    # Planner output
    steps: Optional[str]

    # Retriever output
    retrieved_chunks: Optional[List[str]]
    retrieved_metadata: Optional[List[Dict]]

    # Reasoning output
    reasoning_summary: Optional[str]

    # Final answer
    final_answer: Optional[str]
    confidence: Optional[float]
    hallucinated: Optional[bool]
