import os
import sys
import uuid
import json
from datetime import datetime
from contextvars import ContextVar
from typing import Optional
from loguru import logger

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_DIR = os.getenv("LOG_DIR", "logs")

# The global context holder for async execution chains
correlation_id_ctx: ContextVar[Optional[str]] = ContextVar("correlation_id", default=None)


def generate_correlation_id() -> str:
    return str(uuid.uuid4())


def set_correlation_id(cid: str):
    """Sets the correlation ID for the current context."""
    correlation_id_ctx.set(cid)


def get_current_correlation_id() -> str:
    """Retrieves the correlation ID or fallbacks to a new one if outside context."""
    return correlation_id_ctx.get() or generate_correlation_id()


class JsonLogSink:
    def write(self, message):
        try:
            record = message.record

            log = {
                "timestamp": datetime.utcnow().isoformat(),
                "level": record["level"].name,
                "message": record["message"],
                "logger": record["name"],
                "module": record["module"],
                "function": record["function"],
                "line": record["line"],
            }

            # Inject the current ContextVar correlation_id dynamically if missing from 'extra'
            log["correlation_id"] = record["extra"].get("correlation_id") or correlation_id_ctx.get()

            # Add remaining extra fields
            for k, v in record["extra"].items():
                if k != "correlation_id":
                    log[k] = v

            # Using sys.stdout.write over print avoids the GIL bottlenecks under load
            sys.stdout.write(json.dumps(log) + "\n")

        except Exception as e:
            sys.stderr.write(f"Log formatting error: {e}\n")


def configure_logging():
    logger.remove()

    # Console JSON logs
    logger.add(
        JsonLogSink(),
        level=LOG_LEVEL,
        enqueue=True,
        backtrace=True,
        diagnose=False,
    )

    # Rotating file logs
    os.makedirs(LOG_DIR, exist_ok=True)
    logger.add(
        f"{LOG_DIR}/app.log",
        rotation="10 MB",
        retention="14 days",
        compression="zip",
        level="DEBUG",
        enqueue=True,
        backtrace=True,
        diagnose=False,
        format="{message}",
    )

    return logger


# Configure at execution startup
configure_logging()

__all__ = ["logger", "generate_correlation_id", "correlation_id_ctx", "get_current_correlation_id"]