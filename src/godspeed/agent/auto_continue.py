"""Auto-continue for seamless agent continuation."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class AutoContinue:
    """Auto-continue agent when it pauses.

    Automatically continues agent execution when it would
    otherwise stop, similar to Windsurf's auto-continue feature.
    """

    def __init__(
        self,
        enabled: bool = True,
        max_continues: int = 5,
        auto_threshold: float = 0.8,
    ) -> None:
        self.enabled = enabled
        self.max_continues = max_continues
        self.auto_threshold = auto_threshold
        self.continue_count = 0

    def can_continue(self) -> bool:
        """Check if continuation is allowed."""
        if not self.enabled:
            return False
        if self.continue_count >= self.max_continues:
            logger.info("AutoContinue: max continues reached")
            return False
        return True

    def should_auto_continue(self, agent_output: str) -> bool:
        """Determine if we should auto-continue based on output.

        Looks for signals that the agent is waiting/paused rather than done.
        """
        if not self.can_continue():
            return False

        # Signals that suggest the agent is pausing, not done
        pause_signals = [
            "continue",
            "should i continue",
            "would you like",
            "shall i proceed",
            "is this correct",
            "does this look right",
            "ready to continue",
            "waiting for",
        ]

        output_lower = agent_output.lower()

        for signal in pause_signals:
            if signal in output_lower:
                return True

        return False

    async def continue_agent(self, agent, context: Any) -> Any:
        """Continue agent execution with minimal prompt."""
        self.continue_count += 1
        logger.info(
            "AutoContinue: continuing agent (count=%d/%d)",
            self.continue_count,
            self.max_continues,
        )

        # Inject continue prompt
        continue_prompt = "Continue where you left off."

        from godspeed.agent.loop import run_loop

        result = await run_loop(
            user_input=continue_prompt,
            existing_agent=agent,
            max_iterations=5,
        )

        return result

    def reset(self) -> None:
        """Reset continue count."""
        self.continue_count = 0

    def get_stats(self) -> dict[str, Any]:
        """Get auto-continue stats."""
        return {
            "enabled": self.enabled,
            "continue_count": self.continue_count,
            "max_continues": self.max_continues,
            "remaining": self.max_continues - self.continue_count,
        }


# Default instance
_auto_continue: AutoContinue | None = None


def get_auto_continue() -> AutoContinue:
    """Get or create the auto-continue instance."""
    global _auto_continue
    if _auto_continue is None:
        _auto_continue = AutoContinue()
    return _auto_continue


def should_auto_continue(agent_output: str) -> bool:
    """Check if agent should auto-continue."""
    return get_auto_continue().should_auto_continue(agent_output)


async def continue_if_needed(agent, context: Any) -> Any:
    """Continue agent if output suggests it should continue."""
    ac = get_auto_continue()
    if ac.should_auto_continue(agent.last_output):
        return await ac.continue_agent(agent, context)
    return None
