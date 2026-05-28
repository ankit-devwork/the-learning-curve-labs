import os
import json
import logging
from typing import Dict, Any, List, Sequence, Annotated, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from typing import Literal

# LangGraph Core State, Graph Engine, and Node Architecture Components
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

# Database and External Execution Orchestrators
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import db_service
from litellm import acompletion
from app.services.graph_rag_service import graph_rag_service
from app.core.load_property import settings
from app.observability.logger import logger

# Initialize contextual module logging
logger = logging.getLogger("app.agents.agent_graph")


# -----------------------------------------------------------------------------------------
# 📋 ROUTING DECISION SCHEMA (The Structured Router Envelope)
# -----------------------------------------------------------------------------------------
class RouterDecisionSchema(BaseModel):
    """
    Validates the structured output formatting extracted from the Triage LLM.
    Enforces absolute path predictability, skipping loose string completions.
    """
    reasoning: str = Field(
        description="Brief justification of why this specific data lookup path was chosen based on the conversation history."
    )
    next_step: Literal["hybrid_path", "vector_only_path"] = Field(
        description="Choose 'hybrid_path' if the query evaluates links, connected actors, cross-document entities, or risk networks. Otherwise, choose 'vector_only_path'."
    )
    rephrased_query: str = Field(
        description="The user's input re-written into a fully standalone question that merges all historical pronouns and subject themes into explicit terms. (Anaphora Resolution)."
    )
    extracted_entities: List[str] = Field(
        default=[],
        description="An array of core subjects, nouns, or companies extracted from the rephrased query to query in Neo4j topology. Leave empty if vector_only_path is chosen."
    )


# -----------------------------------------------------------------------------------------
# 🧠 THE RE-ENGINEERED COGNITIVE STATE CHANNELS (Conversation Thread Memory Ledger)
# -----------------------------------------------------------------------------------------
class AgentState(TypedDict):
    """
    The shared state dictionary object passed along the node execution thread.
    Utilizes an annotated message channel to accumulate turn-by-turn conversational history.
    🚀 serialization fix: db_session removed to ensure state remains completely msgpack serializable.
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    query: str                    # The raw, untouched user query input string
    rephrased_query: str          # The contextualized query containing expanded historical targets
    retrieved_context: str         # Consolidated text and relationship string blocks injected into the context window
    routing_decision: str          # Dynamic routing string value consumed by conditional execution vectors
    final_answer: str              # Grounded synthesis block compiled to terminate the graph loop
    extracted_entities: List[str]  # Identified entity components routed to knowledge graph traversals


# -----------------------------------------------------------------------------------------
# 🎛️ COGNITIVE GRAPH NODES (Independent Async Processing Stations)
# -----------------------------------------------------------------------------------------

# Inside app/agents/agent_graph.py -> router_node function

async def router_node(state: AgentState) -> Dict[str, Any]:
    """
    STATION 1: The Traffic Controller, Inquiry Rephraser, and Entity Extractor.
    """
    raw_query = state.get("query", "")
    message_history = state.get("messages", [])
    
    formatted_history = ""
    for msg in message_history[-6:]:  
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        formatted_history += f"{role}: {msg.content}\n"

    logger.info(f"[Graph Node: Router] Evaluating intent layout. History depth: {len(message_history)} messages.")

    try:
        # Formulate a clean serialized string of the validation schema
        schema_dump = json.dumps(RouterDecisionSchema.model_json_schema(), indent=2)

        response = await acompletion(
            model=settings.base_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an elite triage router, context rephraser, and entity extractor for a corporate GraphRAG system.\n\n"
                        "YOUR PRIMARY TASKS:\n"
                        "1. **Analyze Conversation History:** Look at the recent dialogue trail provided.\n"
                        "2. **Resolve Pronouns (Anaphora Resolution):** If the latest user input contains references like 'their deadlines', 'its risks', or 'that task', "
                        "rewrite the statement into a standalone query containing the explicit company names or context records found in the history.\n"
                        "3. **Select Execution Path:** Evaluate if the standalone question needs structural graph tracing ('hybrid_path') or single-document lookups ('vector_only_path').\n"
                        "4. **Extract Entities:** Extract explicit subjects, nouns, or tokens from the rephrased question for Neo4j traversal routines.\n\n"
                        f"You MUST return a JSON object populated with your analysis that exactly matches the properties of this schema structure:\n{schema_dump}"
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"CONVERSATION HISTORY TRAIL:\n{formatted_history or 'No dialogue history recorded yet.'}\n\n"
                        f"LATEST USER INPUT: {raw_query}\n\n"
                        "Generate your routing blueprint matching the json validation layout schema instructions."
                    )
                }
            ],
            api_key=os.getenv("GROQ_API_KEY"),
            response_format={"type": "json_object"}
        )

        cleaned_json_string = response.choices[0].message.content.strip()
        
        decision_data = RouterDecisionSchema.model_validate_json(cleaned_json_string)
        logger.info(f"[Graph Node: Router] Path selection: {decision_data.next_step}. Target query contextualized to: '{decision_data.rephrased_query}'")
        
        return {
            "routing_decision": decision_data.next_step,
            "rephrased_query": decision_data.rephrased_query,
            "extracted_entities": decision_data.extracted_entities
        }
        
    except Exception as e:
        logger.error(f"Routing triage block critically collapsed. Bypassing to default hybrid path: {e}")
        return {
            "routing_decision": "hybrid_path", 
            "rephrased_query": raw_query, 
            "extracted_entities": []
        }

async def hybrid_retrieval_node(state: AgentState) -> Dict[str, Any]:
    """
    STATION 2A: Deep Network Lookups (Postgres pgvector + Neo4j Topology).
    Executes a blended hybrid text and multihop graph retrieval sequence.
    """
    target_query = state.get("rephrased_query") or state.get("query")
    entities = state.get("extracted_entities", [])
    
    logger.info(f"[Graph Node: Hybrid] Invoking joint vector and structural Neo4j lookups for {len(entities)} actors...")
    
    # 🚀 FIX: Localize the transactional session pool boundary inside the execution vertex
    async with db_service.session_factory() as session:
        hybrid_context_matrix = await graph_rag_service.retrieve_hybrid_context(
            query=target_query, 
            db=session,
            entities=entities,
            limit=5
        )

    return {"retrieved_context": hybrid_context_matrix}


async def vector_retrieval_node(state: AgentState) -> Dict[str, Any]:
    """
    STATION 2B: Fast Document Ingestion Lookups (Postgres Vectors Only).
    Runs isolated similarity queries against pgvector tables, bypassing the Neo4j landscape.
    """
    target_query = state.get("rephrased_query") or state.get("query")
    logger.info("[Graph Node: Vector Only] Executing isolated pgvector database segment sweep...")
    
    # 🚀 FIX: Localize the transactional session pool boundary inside the execution vertex
    async with db_service.session_factory() as session:
        raw_text_chunks = await graph_rag_service.retrieve_vector_only_context(
            query=target_query, 
            db=session, 
            limit=5
        )
    
    # Structure text segment lists cleanly into standard context string indicators
    formatted_chunks = ""
    if raw_text_chunks:
        for idx, chunk in enumerate(raw_text_chunks, 1):
            formatted_chunks += f"[{idx}] {chunk.strip()}\n"
    else:
        formatted_chunks = "No semantic textual snippets retrieved for this query footprint."

    context = (
        "=== UNSTRUCTURED SEMANTIC TEXT CHUNKS (VECTOR SPACE ONLY) ===\n"
        f"{formatted_chunks}\n"
        "=== STRUCTURED KNOWLEDGE CONNECTIVITY ===\n"
        "- Structural Neo4j space bypassed via runtime query optimization flags."
    )
    return {"retrieved_context": context}


async def synthesis_node(state: AgentState) -> Dict[str, Any]:
    """
    STATION 3: The Response Synthesis Engine.
    Converts accumulated data layers into grounded, conversational answers.
    """
    logger.info("[Graph Node: Synthesis] Formatting prompt schema with retrieved reference structures...")
    
    target_query = state.get("rephrased_query") or state.get("query")
    gathered_context = state.get("retrieved_context", "")
    message_history = state.get("messages", [])

    system_instruction = (
        "You are an advanced cognitive reasoning engine operating inside a multi-node GraphRAG workspace.\n"
        "Your task is to synthesize a clear, comprehensive, and perfectly grounded answer to the user's inquiry.\n\n"
        "CRITICAL RULES FOR COGNITIVE ALIGNMENT:\n"
        "1. **GROUND YOUR ANSWER:** Base your answer strictly on the injected context matrices. Do not extrapolate facts.\n"
        "2. **Acknowledge Conversation History:** Use the attached history chain to maintain flow and conversational perspective.\n"
        "3. **Incorporate Structural Constraints:** Explicitly mention specific structural linkages, risks, metadata, or tasks found in the text."
    )

    # Convert chat state logs cleanly into structured LLM prompt history frames
    input_payload_messages = [
        {"role": "system", "content": system_instruction}
    ]
    
    for msg in message_history:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        input_payload_messages.append({"role": role, "content": msg.content})
        
    input_payload_messages.append(
        {"role": "user", "content": f"Context Matrix References:\n{gathered_context}\n\nTarget Inquiry: {target_query}"}
    )

    response = await acompletion(
        model=settings.base_model,
        messages=input_payload_messages,
        api_key=os.getenv("GROQ_API_KEY")
    )
    
    compiled_answer = response.choices[0].message.content
    
    return {
        "final_answer": compiled_answer,
        # Append the new conversation turn to update checkpointer message memory channels
        "messages": [HumanMessage(content=state["query"]), AIMessage(content=compiled_answer)]
    }


# -----------------------------------------------------------------------------------------
# 📐 GRAPH COMPILATION ARCHITECTURE (The Execution Graph Blueprint)
# -----------------------------------------------------------------------------------------

workflow = StateGraph(AgentState)

workflow.add_node("router", router_node)
workflow.add_node("hybrid_retrieval", hybrid_retrieval_node)
workflow.add_node("vector_retrieval", vector_retrieval_node)
workflow.add_node("synthesis", synthesis_node)

workflow.set_entry_point("router")

workflow.add_conditional_edges(
    "router",
    lambda state: state["routing_decision"],
    {
        "hybrid_path": "hybrid_retrieval",
        "vector_only_path": "vector_retrieval"
    }
)

workflow.add_edge("hybrid_retrieval", "synthesis")
workflow.add_edge("vector_retrieval", "synthesis")
workflow.add_edge("synthesis", END)

# Module properties written to handle runtime dynamic injection
graph_rag_executor = None
pool = None