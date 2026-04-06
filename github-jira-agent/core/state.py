# core.state.py
import operator
from typing import Annotated, TypedDict, List, Literal, Optional
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    """Agent state with message history and metadata"""
    messages: Annotated[List[BaseMessage], operator.add]
    discovered_file_path: Optional[str]
    jira_issue: Optional[str]
    correlation_id: str 
    branch_name: Optional[str]

