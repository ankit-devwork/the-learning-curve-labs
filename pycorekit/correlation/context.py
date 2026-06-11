from contextvars import ContextVar
import uuid

# Internal context var
_correlation_id_ctx: ContextVar[str] = ContextVar("correlation_id", default=None)

# Backward compatibility for older imports
correlation_id_ctx = _correlation_id_ctx


def generate_correlation_id() -> str:
    """Generate a new correlation ID."""
    return str(uuid.uuid4())


def set_current_correlation_id(cid: str):
    """Set correlation ID into context."""
    _correlation_id_ctx.set(cid)


def get_current_correlation_id() -> str:
    """Retrieve correlation ID from context."""
    return _correlation_id_ctx.get()


def clear_correlation_id():
    """Clear correlation ID after request completes."""
    _correlation_id_ctx.set(None)
