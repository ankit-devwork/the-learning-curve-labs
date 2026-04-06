#config.logger.py
import sys
import uuid
import os
from loguru import logger

def setup_logging():
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)
    
    logger.remove()
    logger.configure(extra={"trace_id": "GLOBAL"}) 

    log_format = (
        "<blue>[{extra[trace_id]}]</blue> "
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )

    logger.add(sys.stderr, format=log_format, level="INFO", colorize=True)
    logger.add(
        "logs/agent_trace.log", 
        rotation="10MB", 
        retention="7 days",
        format=log_format, 
        level="DEBUG"
    )
    return logger

def generate_trace_id():
    """Generates a short, unique ID for session tracking."""
    return str(uuid.uuid4())[:8].upper()