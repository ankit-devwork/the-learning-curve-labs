import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from enum import Enum
from typing import TypeVar

from pycorekit.core_logging.logger import get_logger

from app.core.config import settings
from app.core.exceptions import ServiceUnavailableException

log = get_logger("resilience")

T = TypeVar("T")


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Simple async circuit breaker for external dependencies (LLM, Storage)."""

    def __init__(
        self,
        *,
        name: str,
        failure_threshold: int | None = None,
        recovery_seconds: float | None = None,
    ) -> None:
        self.name = name
        self.failure_threshold = failure_threshold or settings.circuit_breaker_failure_threshold
        self.recovery_seconds = recovery_seconds or settings.circuit_breaker_recovery_sec
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._opened_at: float | None = None

    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN and self._opened_at is not None:
            if time.monotonic() - self._opened_at >= self.recovery_seconds:
                self._state = CircuitState.HALF_OPEN
        return self._state

    def _record_success(self) -> None:
        self._failure_count = 0
        self._opened_at = None
        self._state = CircuitState.CLOSED

    def _record_failure(self) -> None:
        self._failure_count += 1
        if self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN
            self._opened_at = time.monotonic()
            log.warning(
                "Circuit breaker opened",
                breaker=self.name,
                failures=self._failure_count,
            )

    async def call(self, func: Callable[[], Awaitable[T]]) -> T:
        state = self.state
        if state == CircuitState.OPEN:
            raise ServiceUnavailableException(
                f"{self.name} circuit breaker is open — try again later"
            )

        try:
            result = await func()
        except Exception:
            self._record_failure()
            raise

        self._record_success()
        return result


llm_circuit = CircuitBreaker(name="llm")
storage_circuit = CircuitBreaker(name="storage")


def _retryable_exception(exc: BaseException) -> bool:
    if isinstance(exc, ServiceUnavailableException):
        return False
    message = str(exc).lower()
    retry_markers = (
        "timeout",
        "timed out",
        "connection",
        "503",
        "502",
        "504",
        "429",
        "rate limit",
        "overloaded",
        "temporarily unavailable",
    )
    return any(marker in message for marker in retry_markers)


async def with_retry(
    func: Callable[[], Awaitable[T]],
    *,
    operation: str,
    max_attempts: int | None = None,
    base_delay: float | None = None,
    max_delay: float | None = None,
) -> T:
    attempts = max_attempts or settings.retry_max_attempts
    base = base_delay or settings.retry_base_delay_sec
    cap = max_delay or settings.retry_max_delay_sec
    last_exc: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            return await func()
        except Exception as exc:
            last_exc = exc
            if attempt >= attempts or not _retryable_exception(exc):
                raise
            delay = min(cap, base * (2 ** (attempt - 1)))
            delay += random.uniform(0, delay * 0.25)
            log.warning(
                "Retrying operation",
                operation=operation,
                attempt=attempt,
                max_attempts=attempts,
                delay_sec=round(delay, 2),
                error=str(exc),
            )
            await asyncio.sleep(delay)

    assert last_exc is not None
    raise last_exc
