"""Circuit breaker for LLM error recovery."""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """Circuit breaker for API calls.

    Prevents cascading failures by tracking error rates
    and temporarily blocking calls when threshold exceeded.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 3,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self._failure_count = 0
        self._last_failure_time: float | None = None
        self._state = "closed"
        self._half_open_calls = 0

    @property
    def state(self) -> str:
        """Current circuit state."""
        if self._state == "closed":
            return "closed"

        if self._state == "open":
            if self._last_failure_time and time.time() - self._last_failure_time > self.recovery_timeout:
                self._state = "half-open"
                self._half_open_calls = 0
            return "half-open"

        return self._state

    def can_execute(self) -> bool:
        """Check if execution is allowed."""
        state = self.state

        if state == "closed":
            return True

        if state == "half-open":
            return self._half_open_calls < self.half_open_max_calls

        return False

    def record_success(self) -> None:
        """Record a successful call."""
        if self._state == "half-open":
            self._half_open_calls += 1
            if self._half_open_calls >= self.half_open_max_calls:
                self._state = "closed"
                self._failure_count = 0
                logger.info("Circuit breaker closed after successful recovery")
        elif self._state == "closed":
            self._failure_count = max(0, self._failure_count - 1)

    def record_failure(self) -> None:
        """Record a failed call."""
        self._failure_count += 1
        self._last_failure_time = time.time()

        if self._state == "half-open":
            self._state = "open"
            logger.warning("Circuit breaker reopened after half-open failure")
        elif self._failure_count >= self.failure_threshold:
            self._state = "open"
            logger.warning(
                "Circuit breaker opened after %d failures", self._failure_count
            )

    def reset(self) -> None:
        """Reset the circuit breaker."""
        self._failure_count = 0
        self._state = "closed"
        self._half_open_calls = 0
        self._last_failure_time = None


_CIRCUIT_BREAKERS: dict[str, CircuitBreaker] = {}


def get_circuit_breaker(name: str = "default") -> CircuitBreaker:
    """Get or create a circuit breaker."""
    if name not in _CIRCUIT_BREAKERS:
        _CIRCUIT_BREAKERS[name] = CircuitBreaker()
    return _CIRCUIT_BREAKERS[name]


def can_call(name: str = "default") -> bool:
    """Check if an API call is allowed."""
    return get_circuit_breaker(name).can_execute()


def record_success(name: str = "default") -> None:
    """Record a successful call."""
    get_circuit_breaker(name).record_success()


def record_failure(name: str = "default") -> None:
    """Record a failed call."""
    get_circuit_breaker(name).record_failure()


def get_status(name: str = "default") -> dict[str, Any]:
    """Get circuit breaker status."""
    cb = get_circuit_breaker(name)
    return {"state": cb.state, "failure_count": cb._failure_count}