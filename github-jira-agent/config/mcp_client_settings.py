# config.mcp-client-settings.py
import os
from dotenv import load_dotenv
from config.logger import logger

load_dotenv()


LLM_TIMEOUT = 30  # seconds
TOOL_TIMEOUT = 10  # seconds
MAX_ITERATIONS = 10


def validate_env():
    REQUIRED_ENV_VARS = [
        "GITHUB_TOKEN",
        "OPENAI_API_KEY",
        "JIRA_BASE_URL",
        "JIRA_EMAIL",
        "JIRA_API_TOKEN"
    ]
    
    # Check for trace_id context, default to SETUP if not present
    missing_vars = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]
    
    if missing_vars:
        # We use .bind here because this runs before the graph generates a session ID
        logger.bind(trace_id="SETUP").error(f"❌ Missing: {', '.join(missing_vars)}")
        return False
    
    logger.bind(trace_id="SETUP").info("✅ Environment validated.")
    return True

# Constants for easy modification
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
REPO_OWNER = "ankit-devwork"
REPO_NAME = "springboot-demo"

SERVERS = {
    "github": {
        "transport": "stdio", 
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-github"],
        "env": {
            "GITHUB_PERSONAL_ACCESS_TOKEN": GITHUB_TOKEN,
            "GITHUB_TOKEN": GITHUB_TOKEN,
            **os.environ
        }
    },
    "jira": {
        "transport": "stdio", 
        "command": "npx",
        "args": ["-y", "@nexus2520/jira-mcp-server"],
        "env": {
            "JIRA_BASE_URL": os.getenv("JIRA_BASE_URL"),
            "JIRA_EMAIL": os.getenv("JIRA_EMAIL"),
            "JIRA_API_TOKEN": os.getenv("JIRA_API_TOKEN"),
            **os.environ
        }
    }
}