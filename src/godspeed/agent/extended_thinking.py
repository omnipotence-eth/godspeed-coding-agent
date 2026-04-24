"""Extended thinking for complex reasoning."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class ExtendedThinking:
    """Enable deeper reasoning for complex problems.

    Provides a structured thinking process that breaks down
    complex problems into manageable steps.
    """

    @staticmethod
    def think(
        problem: str,
        context: dict[str, Any] | None = None,
        effort: str = "medium",
    ) -> str:
        """Generate a thinking trace for the problem.

        Args:
            problem: The complex problem description
            context: Any relevant context
            effort: "low", "medium", or "high"

        Returns:
            A structured thinking trace
        """
        effort_config = {
            "low": {"max_steps": 3, "depth": "surface"},
            "medium": {"max_steps": 7, "depth": "moderate"},
            "high": {"max_steps": 15, "depth": "deep"},
        }

        config = effort_config.get(effort, effort_config["medium"])
        max_steps = config["max_steps"]
        depth = config["depth"]

        trace = [
            f"## Extended Thinking (effort: {effort}, depth: {depth})",
            f"### Problem\n{problem}",
        ]

        if context:
            trace.append(f"### Context")
            for k, v in context.items():
                trace.append(f"- {k}: {v}")

        # Generate reasoning steps
        steps = _generate_reasoning_steps(problem, max_steps, depth)
        trace.extend(steps)

        return "\n".join(trace)


def _generate_reasoning_steps(
    problem: str, max_steps: int, depth: str
) -> list[str]:
    """Generate reasoning steps for the problem."""
    steps = ["### Reasoning Steps"]

    # Step 1: Understand the problem
    steps.append("1. **Understand**: Break down the problem into core requirements")

    # Step 2: Analyze constraints
    steps.append("2. **Analyze**: Identify constraints, dependencies, and edge cases")

    # Step 3: Plan approach
    steps.append("3. **Plan**: Design approach and identify files to modify")

    if depth in ("moderate", "deep"):
        # Step 4: Consider alternatives
        steps.append("4. **Alternative**: Consider alternative approaches")

        # Step 5: Evaluate tradeoffs
        steps.append("5. **Tradeoff**: Evaluate time/complexity tradeoffs")

    if depth == "deep":
        # Step 6: Risk assessment
        steps.append("6. **Risk**: Identify potential issues and mitigations")

        # Step 7: Final plan
        steps.append("7. **Finalize**: Produce final implementation plan")

    return steps[:max_steps]


_extended_thinking_prompt = """\
You are performing extended reasoning on this problem. Take your time to think through it carefully.

## Guidelines
- Break down the problem into smaller parts
- Consider edge cases and error conditions
- Think about how changes might affect other parts of the system
- Plan for testing and verification

## Output
Provide your reasoning in a structured format.
"""


def get_extended_thinking_prompt() -> str:
    """Get the system prompt for extended thinking."""
    return _extended_thinking_prompt


# Usage tracking
_usage: dict[str, int] = {"low": 0, "medium": 0, "high": 0}


def record_usage(effort: str) -> None:
    """Record extended thinking usage."""
    global _usage
    if effort in _usage:
        _usage[effort] += 1


def get_usage_stats() -> dict[str, int]:
    """Get usage statistics."""
    return _usage.copy()