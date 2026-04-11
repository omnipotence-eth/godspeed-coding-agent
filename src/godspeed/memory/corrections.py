"""Correction tracker — detects and records user corrections for learning.

Heuristic: when a user message follows an agent action and contains negation
patterns ("no", "don't", "stop", "instead") plus an instruction, it's likely
a correction. Corrections are stored in UserMemory and surfaced in the system
prompt as "User prefers X over Y" guidance.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from godspeed.memory.user_memory import UserMemory

logger = logging.getLogger(__name__)

# Patterns that suggest the user is correcting agent behavior
_NEGATION_PATTERNS = [
    r"\bno\b",
    r"\bdon'?t\b",
    r"\bstop\b",
    r"\bnot\b",
    r"\binstead\b",
    r"\bwrong\b",
    r"\bactually\b",
    r"\bplease\s+(?:don'?t|stop|use)\b",
    r"\bnever\b",
    r"\balways\b",
    r"\bprefer\b",
]

_NEGATION_RE = re.compile("|".join(_NEGATION_PATTERNS), re.IGNORECASE)


def is_likely_correction(user_message: str) -> bool:
    """Detect whether a user message is likely a correction.

    A correction typically contains negation/preference language and is
    directive in nature (short to medium length, imperative).
    """
    stripped = user_message.strip()
    if not stripped:
        return False

    # Too short or too long messages are unlikely corrections
    word_count = len(stripped.split())
    if word_count < 2 or word_count > 100:
        return False

    return bool(_NEGATION_RE.search(stripped))


class CorrectionTracker:
    """Tracks user corrections and provides them for system prompt injection.

    Works with UserMemory to persist corrections across sessions.
    The agent loop calls `check_for_correction()` after each user message
    that follows an agent action.
    """

    def __init__(self, user_memory: UserMemory) -> None:
        self._memory = user_memory

    def check_for_correction(
        self,
        user_message: str,
        last_agent_action: str = "",
    ) -> int | None:
        """Check if user_message is a correction and record it if so.

        Args:
            user_message: The user's message to check.
            last_agent_action: Description of the last agent action (tool call, etc.).

        Returns:
            The correction ID if recorded, None otherwise.
        """
        if not is_likely_correction(user_message):
            return None

        correction_id = self._memory.record_correction(
            original=last_agent_action or "(agent action)",
            corrected=user_message,
            context="auto-detected",
        )
        logger.info(
            "correction_tracker.detected id=%d message=%s",
            correction_id,
            user_message[:80],
        )
        return correction_id

    def get_top_corrections(self, n: int = 5) -> list[dict[str, Any]]:
        """Get the top N most recent corrections for system prompt injection.

        Returns dicts with 'original' and 'corrected' keys.
        """
        return self._memory.get_corrections(limit=n)

    def format_for_system_prompt(self, n: int = 5) -> str:
        """Format top corrections as system prompt guidance.

        Returns a string like:
            User corrections:
            - User prefers: "use logger" over "use print"
            - User prefers: "snake_case" over "camelCase"
        """
        corrections = self.get_top_corrections(n)
        if not corrections:
            return ""

        lines = ["User corrections (learn from these):"]
        for c in corrections:
            original = c["original"][:60]
            corrected = c["corrected"][:60]
            lines.append(f'- User said: "{corrected}" (after: "{original}")')
        return "\n".join(lines)
