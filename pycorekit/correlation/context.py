"""
Correlation ID context management using ContextVar.
"""

import uuid
from contextvars import ContextVar

correlation_id_ctx = ContextVar("correlation_id", default=None)

def generate_correlation_id():
    return str(uuid.uuid4())

def set_correlation_id(cid: str):
    correlation_id_ctx.set(cid)

def get_current_correlation_id():
    return correlation_id_ctx.get()
