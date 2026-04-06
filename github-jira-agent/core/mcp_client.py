# core.mcp_client.py
import asyncio
from config.logger import logger
from langchain_mcp_adapters.client import MultiServerMCPClient
from config.mcp_client_settings import TOOL_TIMEOUT,SERVERS

# Initialize the client
mcp_client = MultiServerMCPClient(SERVERS)

class ToolManager:
    """Manages MCP tools with caching and safe execution."""
    _cache = None

    @classmethod
    async def get_tools(cls):
        """Get tools from MCP with a singleton-style cache."""
        if cls._cache is None:
            # We bind 'SETUP' because this usually happens at boot
            logger.bind(trace_id="SETUP").info("📡 Connecting to MCP Servers (GitHub/Jira)...")
            try:
                tools = await mcp_client.get_tools()
                cls._cache = {t.name: t for t in tools}
                logger.bind(trace_id="SETUP").success(f"✅ Cached {len(cls._cache)} tools")
            except Exception as e:
                logger.bind(trace_id="SETUP").error(f"❌ Failed to fetch tools: {e}")
                raise
        return cls._cache

async def execute_tool(tool_name: str, tool_args: dict, named_tools: dict, trace_id: str) -> str:
    """Execute a tool with full correlation tracking and timeouts."""
    # This ensures EVERY log inside this function has the [ID]
    with logger.contextualize(trace_id=trace_id):
        try:
            tool = named_tools.get(tool_name)
            if not tool:
                logger.error(f"Tool '{tool_name}' not found in registry")
                return f"Error: Tool '{tool_name}' not found"

            logger.info(f"🛠️  Executing: {tool_name} with args: {tool_args}")
            result = await asyncio.wait_for(
                tool.ainvoke(tool_args),
                timeout=TOOL_TIMEOUT
            )

            logger.success(f"✅ {tool_name} returned data successfully")
            return str(result)

        except asyncio.TimeoutError:
            logger.warning(f"⏱️ Tool timeout after {TOOL_TIMEOUT}s")
            return "Error: Request timed out."
        except Exception as e:
            logger.error(f"⚠️ {type(e).__name__}: {str(e)}")
            return f"Error: {str(e)}"