"""Track repeated user approvals to suggest auto-permission rules."""

from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)


class ApprovalTracker:
    """Session-scoped tracker for repeated permission approvals.

    When a user approves the same tool pattern multiple times, this tracker
    can suggest adding it as a permanent allow rule.

    Thread-safe — all mutations are guarded by a lock.
    """

    def __init__(self, threshold: int = 3) -> None:
        self._threshold = threshold
        self._counts: dict[str, int] = {}
        self._suggested: set[str] = set()
        self._lock = threading.Lock()

    def record_approval(self, pattern: str) -> None:
        """Record a user approval for a tool pattern."""
        with self._lock:
            self._counts[pattern] = self._counts.get(pattern, 0) + 1
            logger.debug(
                "Approval recorded pattern=%s count=%d",
                pattern,
                self._counts[pattern],
            )

    def should_suggest(self, pattern: str, threshold: int | None = None) -> bool:
        """Check if we should suggest adding this pattern as a permanent rule.

        Returns True exactly once per pattern when the threshold is reached.
        """
        effective_threshold = threshold if threshold is not None else self._threshold
        with self._lock:
            count = self._counts.get(pattern, 0)
            if count >= effective_threshold and pattern not in self._suggested:
                self._suggested.add(pattern)
                logger.info(
                    "Suggesting auto-permission for pattern=%s after %d approvals",
                    pattern,
                    count,
                )
                return True
            return False

    def get_count(self, pattern: str) -> int:
        """Get the current approval count for a pattern."""
        with self._lock:
            return self._counts.get(pattern, 0)

    def reset(self) -> None:
        """Clear all tracked approvals and suggestions."""
        with self._lock:
            self._counts.clear()
            self._suggested.clear()
