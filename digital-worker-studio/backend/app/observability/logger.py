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

# Global context holder for async execution chains
correlation_id_ctx: ContextVar[Optional[str]] = ContextVar("correlation_id", default=None)


def generate_correlation_id() -> str:
    return str(uuid.uuid4())


def set_correlation_id(cid: str):
    correlation_id_ctx.set(cid)


def get_current_correlation_id() -> Optional[str]:
    return correlation_id_ctx.get()


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

            # IMPORTANT: do NOT generate a new correlation_id here
            log["correlation_id"] = record["extra"].get("correlation_id")

            # Add remaining extra fields
            for k, v in record["extra"].items():
                if k == "correlation_id":
                    continue
                try:
                    json.dumps({k: v})
                    log[k] = v
                except Exception:
                    log[k] = str(v)

            sys.stdout.write(json.dumps(log) + "\n")

        except Exception as e:
            sys.stderr.write(f"Log formatting error: {e}\n")


def configure_logging():
    logger.remove()

    logger.add(
        JsonLogSink(),
        level=LOG_LEVEL,
        enqueue=True,
        backtrace=True,
        diagnose=False,
    )

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


configure_logging()

__all__ = ["logger", "generate_correlation_id", "correlation_id_ctx", "get_current_correlation_id", "set_correlation_id"]


def get_bound_logger(**extra):
    cid = get_current_correlation_id()
    return logger.bind(correlation_id=cid, **extra)


def get_request_logger(request, **extra):
    cid = getattr(request.state, "correlation_id", None)
    return logger.bind(correlation_id=cid, **extra)


__all__.append("get_bound_logger")
__all__.append("get_request_logger")
