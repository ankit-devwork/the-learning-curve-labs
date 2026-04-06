import asyncio
import os
import re
import uuid
from typing import Literal, List, Dict, Any, Tuple

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    ToolMessage,
    SystemMessage
)
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from config.logger import logger
from config.mcp_client_settings import LLM_TIMEOUT, MAX_ITERATIONS
from core.mcp_client import ToolManager, execute_tool
from core.state import AgentState

load_dotenv()

GITHUB_OWNER = os.getenv("GITHUB_OWNER", "ankit-devwork")
GITHUB_REPO = os.getenv("GITHUB_REPO", "springboot-demo")
GITHUB_BASE_BRANCH = os.getenv("GITHUB_BASE_BRANCH", "main")

LLM_NAME="gpt-4o-mini"

# ═══════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

async def categorize_tools_by_safety(named_tools: Dict[str, Any]) -> Tuple[Dict, Dict]:
    """Categorize tools by safety level"""
    safe_keywords = ["read", "get", "list", "search", "fetch", "view", "show", "describe"]
    exclude_from_suggestion = [
        "fork_repository", "add_issue_comment", "create_pull_request_review",
        "merge_pull_request", "update_pull_request_branch", "create_pull_request",
        "create_branch", "create_or_update_file", "push_files"
    ]
    
    safe_tools, write_tools = {}, {}
    for name, tool in named_tools.items():
        if any(excluded in name.lower() for excluded in exclude_from_suggestion):
            write_tools[name] = tool
            continue
        description = (tool.description or "").lower()
        if any(kw in description for kw in ["create", "update", "delete", "merge"]):
            write_tools[name] = tool
        else:
            safe_tools[name] = tool
    return safe_tools, write_tools


# ✅ CRITICAL: Tool argument validation
def validate_tool_args(tool_name: str, args: dict) -> tuple:
    """
    Validate and fix tool arguments.
    ✅ Called in EVERY tool execution loop
    """
    
    # Define expected parameters for each tool
    expected_params = {
        "get_file_contents": ["owner", "repo", "path", "branch"],
        "create_branch": ["owner", "repo", "branch"],
        "create_or_update_file": ["owner", "repo", "path", "content", "message", "branch", "sha"],
        "create_pull_request": ["owner", "repo", "title", "body", "head", "base"],
        "merge_pull_request": ["owner", "repo", "pull_number"],
        "search_code": ["q", "per_page"],
        "jira_search": ["action", "jql"],
    }
    
    if tool_name not in expected_params:
        return True, ""  # Unknown tool, skip validation
    
    # ✅ FIX: Common typos
    typo_map = {
        "ppath": "path",
        "brancch": "branch",
        "pul_number": "pull_number",
    }
    
    # Check for typos and fix them
    for typo, correct in typo_map.items():
        if typo in args:
            logger.warning(f"🔧 Fixed typo in {tool_name}: '{typo}' → '{correct}'")
            args[correct] = args.pop(typo)
    
    # ✅ FIX: Extra quotes in string parameters
    for key in list(args.keys()):
        if isinstance(args[key], str):
            if args[key].endswith("''"):
                logger.warning(f"🔧 Removed extra quote from {key}")
                args[key] = args[key][:-1]
    
    return True, ""


async def get_tool_descriptions(named_tools: Dict) -> str:
    """Generate detailed tool descriptions for LLM"""
    
    descriptions = []
    for name, tool in named_tools.items():
        args = tool.args if hasattr(tool, 'args') else {}
        
        if isinstance(args, dict) and args:
            param_str = ", ".join(f"{k}: {v.get('type', 'string')}" for k, v in args.items())
        else:
            param_str = "..."
        
        desc = tool.description if hasattr(tool, 'description') else "No description"
        
        descriptions.append(
            f"- **{name}**({param_str})\n"
            f"  {desc}\n"
        )
    
    return "\n".join(descriptions)


def extract_implementation_from_messages(messages: List) -> Dict[str, str]:
    """Extract implementation details from conversation"""
    details = {
        "title": "requested changes",
        "description": "Implementation of requested feature",
        "file_path": None
    }
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            content = msg.content
            if "file" in content.lower():
                match = re.search(r'file(?:\s+path)?:\s*([^\n]+)', content, re.IGNORECASE)
                if match:
                    details["file_path"] = match.group(1).strip()
            if "endpoint" in content.lower():
                match = re.search(r'endpoint[s]?[:\s]+([^\n.]+)', content, re.IGNORECASE)
                if match:
                    details["title"] = match.group(1).strip()
    return details


# ═══════════════════════════════════════════════════════════════════════════
# NODES
# ═══════════════════════════════════════════════════════════════════════════

async def suggest_node(state: AgentState):
    """Suggestion node with full validation and loop detection"""
    logger.info("\n📍 [Node] Suggestor: Gathering context...")
    
    # Check if routing command
    if state["messages"] and isinstance(state["messages"][-1], HumanMessage):
        content = state["messages"][-1].content.strip().upper()
        if content in ["APPROVE", "MERGE", "ABORT"]:
            logger.debug(f"   Routing command detected: {content}")
            return {"messages": []}
    
    if "correlation_id" not in state:
        state["correlation_id"] = str(uuid.uuid4())
    
    named_tools = await ToolManager.get_tools()
    safe_tools_dict, _ = await categorize_tools_by_safety(named_tools)
    
    tool_schemas = await get_tool_descriptions(safe_tools_dict)
    
    system_prompt = f"""You are a senior developer. Suggest code changes based on requirements.

IMPORTANT - Tool Parameters (use EXACT names):
{tool_schemas}

⚠️ CRITICAL: Use EXACT parameter names:
- Use 'path' (NOT 'ppath')
- Use 'branch' (NOT 'brancch')
- Use 'pull_number' (NOT 'pul_number')

Available tools and their parameters are listed above. Use them exactly as specified."""
    
    llm = ChatOpenAI(model=LLM_NAME, temperature=0).bind_tools(list(safe_tools_dict.values()))
    
    messages = [SystemMessage(content=system_prompt)] + list(state["messages"])
    iteration = 0
    new_msgs = []
    
    # ✅ Track tool calls to detect loops
    tool_call_history = []
    
    while iteration < MAX_ITERATIONS:
        iteration += 1
        logger.debug(f"   [Iteration {iteration}/{MAX_ITERATIONS}]")
        
        try:
            response = await asyncio.wait_for(
                llm.ainvoke(messages),
                timeout=LLM_TIMEOUT
            )
        except asyncio.TimeoutError:
            logger.error(f"⏱️ LLM timeout")
            break
        except Exception as e:
            logger.error(f"❌ LLM error: {e}")
            break
        
        messages.append(response)
        new_msgs.append(response)
        
        if not response.tool_calls:
            logger.info("✅ Suggestion complete")
            break
        
        # ✅ CRITICAL: Detect infinite loops BEFORE executing tools
        current_tools = [tc["name"] for tc in response.tool_calls]
        
        if len(tool_call_history) > 2:
            recent = [t for group in tool_call_history[-3:] for t in group]
            # Check if same tool called 3+ times
            if len(set(recent)) == 1 and recent[0] == current_tools[0]:
                logger.warning(f"🔄 Loop detected: {current_tools[0]} called repeatedly")
                logger.warning("   Breaking to prevent infinite loop")
                
                # ✅ CRITICAL: Create ToolMessages for tools that weren't executed
                for tool_call in response.tool_calls:
                    t_msg = ToolMessage(
                        tool_call_id=tool_call["id"],
                        content="Loop detected - analysis complete to prevent infinite recursion"
                    )
                    messages.append(t_msg)
                    new_msgs.append(t_msg)
                
                break  # ✅ Now safe to break - all tool_calls have responses
        
        tool_call_history.append(current_tools)
        
        # ✅ Execute tools with validation
        for tool_call in response.tool_calls:
            logger.info(f"🔧 Executing tool: {tool_call['name']}")
            
            # ✅ CRITICAL: Validate and fix arguments
            is_valid, error_msg = validate_tool_args(tool_call["name"], tool_call["args"])
            
            if not is_valid:
                logger.error(f"   Invalid args: {error_msg}")
                result = f"Error: Invalid arguments - {error_msg}"
            else:
                try:
                    result = await execute_tool(
                        tool_call["name"],
                        tool_call["args"],
                        named_tools,
                        state.get("correlation_id", "id")
                    )
                    logger.debug(f"   ✅ Tool succeeded")
                except Exception as e:
                    logger.error(f"   ❌ Tool error: {e}")
                    result = f"Error: {str(e)}"
            
            # ✅ CRITICAL: Always append ToolMessage (even if error)
            t_msg = ToolMessage(tool_call_id=tool_call["id"], content=str(result))
            messages.append(t_msg)
            new_msgs.append(t_msg)
    
    return {"messages": new_msgs}


async def create_pr_node(state: AgentState):
    """PR creation/update node with full validation"""
    logger.info("\n📍 [Node] PR Generator: Creating/Updating PR...")
    
    named_tools = await ToolManager.get_tools()
    _, write_tools_dict = await categorize_tools_by_safety(named_tools)
    
    tool_schemas = await get_tool_descriptions(write_tools_dict)
    
    system_prompt = f"""You are a senior developer executing pull request operations.

Available tools:
{tool_schemas}

⚠️ CRITICAL: Parameter Names:
- Use 'path' (NOT 'ppath')
- Use 'branch' (NOT 'brancch')
- Use 'pull_number' (NOT 'pul_number')

Follow instructions exactly."""
    
    llm = ChatOpenAI(model=LLM_NAME, temperature=0).bind_tools(list(write_tools_dict.values()))
    
    jira_issue = state.get("jira_issue") or "SCRUM-6"
    branch_name = state.get("branch_name") or f"fix-{jira_issue.lower()}"
    
    # Check if PR already exists
    existing_pr_match = None
    for msg in reversed(state["messages"]):
        m = re.search(r'/pull/(\d+)', str(msg.content))
        if m:
            existing_pr_match = m.group(0)
            break
    
    directive = f"""Execute these steps in order:
1. Ensure branch '{branch_name}' exists (create if needed, ignore if exists)
2. Update the code file with the latest implementation
"""
    if not existing_pr_match:
        directive += f"3. Create a NEW Pull Request to '{GITHUB_BASE_BRANCH}' from '{branch_name}'"
    else:
        directive += f"3. The PR {existing_pr_match} already exists. Your file update will reflect there automatically."
    
    working_messages = [SystemMessage(content=system_prompt)] + state["messages"][-5:] + [SystemMessage(content=directive)]
    new_msgs = []
    pr_url = None
    
    for i in range(5):
        try:
            response = await asyncio.wait_for(
                llm.ainvoke(working_messages),
                timeout=30
            )
        except asyncio.TimeoutError:
            logger.error("LLM timeout")
            break
        except Exception as e:
            logger.error(f"LLM error: {e}")
            break
        
        working_messages.append(response)
        new_msgs.append(response)
        
        if not response.tool_calls:
            break
        
        # ✅ Execute tools with validation
        for tool_call in response.tool_calls:
            logger.info(f"🚀 Executing: {tool_call['name']}")
            
            # ✅ CRITICAL: Validate and fix arguments
            is_valid, error_msg = validate_tool_args(tool_call["name"], tool_call["args"])
            
            if not is_valid:
                result = f"Error: Invalid arguments"
            else:
                try:
                    result = await execute_tool(
                        tool_call["name"],
                        tool_call["args"],
                        named_tools,
                        state.get("correlation_id", "id")
                    )
                    logger.info(f"   ✅ {tool_call['name']} succeeded")
                    
                    # ✅ Extract PR number if create_pull_request
                    if tool_call["name"] == "create_pull_request":
                        match = re.search(r'pull[/_]?(\d+)', str(result), re.IGNORECASE)
                        if match:
                            pr_number = match.group(1)
                            pr_url = f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}/pull/{pr_number}"
                            logger.info(f"   📎 PR URL: {pr_url}")
                
                except Exception as e:
                    logger.error(f"Tool error: {e}")
                    result = f"Error: {type(e).__name__}: {str(e)[:100]}"
            
            # ✅ CRITICAL: Always append ToolMessage
            t_msg = ToolMessage(tool_call_id=tool_call["id"], content=str(result))
            working_messages.append(t_msg)
            new_msgs.append(t_msg)
    
    # ✅ Build summary with proper PR link
    summary_content = f"✅ **PR {'Updated' if existing_pr_match else 'Created'}!**\n"
    
    if pr_url:
        summary_content += f"PR Link: [{pr_url}]({pr_url})\n"
    elif existing_pr_match:
        summary_content += f"PR Link: {existing_pr_match}\n"
    else:
        pr_link = f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}/pulls"
        summary_content += f"PR Link: [{pr_link}]({pr_link})\n"
    
    summary = AIMessage(content=summary_content)
    return {"messages": new_msgs + [summary]}


async def merge_node(state: AgentState):
    """PR merge node with detailed error handling"""
    logger.info("\n📍 [Node] Merger: Finalizing PR...")
    
    named_tools = await ToolManager.get_tools()
    
    # Find PR number
    pr_number = None
    for msg in reversed(state["messages"]):
        match = re.search(r'/pull/(\d+)', str(msg.content))
        if match:
            pr_number = int(match.group(1))
            break
    
    if not pr_number:
        logger.error("No PR number found")
        return {
            "messages": [AIMessage(
                content="❌ Could not find PR number to merge. "
                        "Please check the conversation history."
            )]
        }
    
    try:
        result = await execute_tool(
            "merge_pull_request",
            {
                "owner": GITHUB_OWNER,
                "repo": GITHUB_REPO,
                "pull_number": pr_number
            },
            named_tools,
            state.get("correlation_id", "id")
        )
        
        if "error" in str(result).lower():
            logger.error(f"Merge result has error: {result}")
            return {
                "messages": [AIMessage(
                    content=f"❌ Merge failed:\n\n{result}\n\n"
                            "This may be due to:\n"
                            "- PR already merged\n"
                            "- Merge conflicts\n"
                            "- Insufficient permissions"
                )]
            }
        
        logger.info(f"✅ PR merged successfully")
        return {
            "messages": [AIMessage(
                content=f"🚀 **Success!** PR #{pr_number} merged into `{GITHUB_BASE_BRANCH}`"
            )]
        }
    
    except Exception as e:
        logger.error(f"Merge error: {type(e).__name__}: {e}", exc_info=True)
        return {
            "messages": [AIMessage(
                content=f"❌ Merge failed: {type(e).__name__}\n\n"
                        f"Error: {str(e)[:200]}\n\n"
                        f"Please check the PR manually at GitHub."
            )]
        }


# ═══════════════════════════════════════════════════════════════════════════
# ROUTING LOGIC
# ═══════════════════════════════════════════════════════════════════════════

def suggestor_router(state: AgentState) -> Literal["pr_generator", "suggestor", "__end__"]:
    """Route from suggestor node"""
    if not state["messages"]:
        return "suggestor"
    
    content = state["messages"][-1].content.strip().upper()
    
    if "APPROVE" in content:
        logger.info("→ Route: suggestor → pr_generator")
        return "pr_generator"
    if "ABORT" in content:
        logger.info("→ Route: suggestor → END")
        return "__end__"
    
    logger.info("→ Route: suggestor → suggestor (continue)")
    return "suggestor"


def pr_router(state: AgentState) -> Literal["merger", "suggestor", "__end__", "pr_generator"]:
    """Route from pr_generator node"""
    if not state["messages"]:
        return "__end__"
    
    content = state["messages"][-1].content.strip().upper()
    
    if "MERGE" in content:
        logger.info("→ Route: pr_generator → merger")
        return "merger"
    if "APPROVE" in content:
        logger.info("→ Route: pr_generator → pr_generator (re-execute)")
        return "pr_generator"
    if "ABORT" in content:
        logger.info("→ Route: pr_generator → END")
        return "__end__"
    
    logger.info("→ Route: pr_generator → suggestor (feedback)")
    return "suggestor"


# ═══════════════════════════════════════════════════════════════════════════
# BUILD GRAPH
# ═══════════════════════════════════════════════════════════════════════════

logger.info("🔨 Building LangGraph...")

builder = StateGraph(AgentState)
builder.add_node("suggestor", suggest_node)
builder.add_node("pr_generator", create_pr_node)
builder.add_node("merger", merge_node)

builder.set_entry_point("suggestor")

builder.add_conditional_edges("suggestor", suggestor_router, {
    "pr_generator": "pr_generator",
    "suggestor": "suggestor",
    "__end__": END
})

builder.add_conditional_edges("pr_generator", pr_router, {
    "merger": "merger",
    "suggestor": "suggestor",
    "pr_generator": "pr_generator",
    "__end__": END
})

builder.add_edge("merger", END)

memory = MemorySaver()
graph = builder.compile(checkpointer=memory, interrupt_after=["suggestor", "pr_generator"])

logger.info("✅ Graph built successfully")