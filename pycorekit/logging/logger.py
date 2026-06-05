"""
Generic Loguru logger initializer with:
- Console sink
- Rotating file sink
- JSON structured log sink
- Correlation ID binding
"""

import os
from loguru import logger
from typing import Optional
from pycorekit.correlation.context import get_current_correlation_id


def init_logger(
    log_dir: str = "logs",
    rotation: str = "00:00",
    retention: str = "7 days",
    compression: str = "zip",
    level: str = "INFO",
    console: bool = True,
    file: bool = True,
    json_file: bool = False,
    format: str = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
):
    """
    Initialize Loguru logger with multiple sinks.
    """

    os.makedirs(log_dir, exist_ok=True)
    logger.remove()

    # Console sink
    if console:
        logger.add(
            sink=lambda msg: print(msg, end=""),
            level=level,
            format=format,
            enqueue=True,
        )

    # File sink
    if file:
        logger.add(
            f"{log_dir}/app_{{time:YYYY-MM-DD}}.log",
            rotation=rotation,
            retention=retention,
            compression=compression,
            level=level,
            enqueue=True,
            backtrace=True,
            diagnose=False,
            format=format,
        )

    # JSON structured logs
    if json_file:
        logger.add(
            f"{log_dir}/json_{{time:YYYY-MM-DD}}.log",
            rotation=rotation,
            retention=retention,
            compression=compression,
            level=level,
            serialize=True,
            enqueue=True,
        )

    return logger


def get_logger(name: Optional[str] = None):
    """
    Returns a logger bound with module name and correlation ID.
    """
    cid = get_current_correlation_id()
    if cid:
        return logger.bind(module=name, correlation_id=cid)
    return logger.bind(module=name)
