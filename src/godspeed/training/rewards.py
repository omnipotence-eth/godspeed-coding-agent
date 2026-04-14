"""Per-step reward annotations for GRPO/DPO fine-tuning.

Computes automatic reward signals for each tool execution step in a
conversation. These rewards enable reinforcement learning (GRPO) on top
of SFT — AgentQ showed RL adds 10-15% over SFT alone.

Rewards are appended to conversation JSONL as ``role: "reward"`` entries
by the ``ConversationLogger``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Canonical efficient tool sequences (order matters)
_EFFICIENT_SEQUENCES: list[tuple[str, ...]] = [
    ("grep_search", "file_read", "file_edit"),
    ("glob_search", "file_read", "file_edit"),
    ("grep_search", "file_read", "file_write"),
    ("file_read", "file_edit"),
    ("web_search", "web_fetch"),
]


@dataclass(frozen=True, slots=True)
class StepReward:
    """Reward annotation for a single tool execution step."""

    step: int
    tool_name: str
    reward: float
    components: dict[str, float]
    reasoning: str


def compute_tool_reward(
    tool_name: str,
    arguments: dict[str, Any],
    is_error: bool,
    permission_denied: bool = False,
    verify_passed: bool | None = None,
    was_retried: bool = False,
    is_dangerous: bool = False,
    step: int = 0,
) -> StepReward:
    """Compute automatic reward for a single tool execution step.

    Args:
        tool_name: Name of the tool executed.
        arguments: Tool call arguments.
        is_error: Whether the tool returned an error.
        permission_denied: Whether the permission engine blocked execution.
        verify_passed: Whether auto-verify passed (None if not applicable).
        was_retried: Whether this was a retry of a previously failed call.
        is_dangerous: Whether the command was flagged as dangerous.
        step: Step number in the conversation.

    Returns:
        StepReward with computed reward and component breakdown.
    """
    components: dict[str, float] = {}
    reasons: list[str] = []

    # Base outcome
    if permission_denied:
        components["permission_denied"] = -0.5
        reasons.append("Permission denied")
    elif is_error:
        components["execution_failed"] = -0.5
        reasons.append("Tool execution failed")
    else:
        components["execution_success"] = 1.0
        reasons.append("Tool executed successfully")

    # Dangerous command penalty
    if is_dangerous:
        components["dangerous_command"] = -1.0
        reasons.append("Dangerous command attempted")

    # Verify bonus
    if verify_passed is True:
        if was_retried:
            components["verify_retry_fixed"] = 0.25
            reasons.append("Verify passed after retry (self-correction)")
        else:
            components["verify_passed"] = 0.5
            reasons.append("Verify passed on first try")
    elif verify_passed is False:
        components["verify_failed"] = -0.25
        reasons.append("Verify found issues")

    total = sum(components.values())

    return StepReward(
        step=step,
        tool_name=tool_name,
        reward=round(total, 2),
        components=components,
        reasoning="; ".join(reasons),
    )


def compute_sequence_bonus(
    tool_sequence: list[str],
) -> float:
    """Compute a bonus reward for efficient tool sequences.

    Checks if the tool sequence matches (or contains) a known efficient
    pattern like grep → read → edit.

    Returns:
        Bonus reward (0.0 if no efficient pattern found, 0.5 if matched).
    """
    if len(tool_sequence) < 2:
        return 0.0

    for pattern in _EFFICIENT_SEQUENCES:
        if _contains_subsequence(tool_sequence, pattern):
            return 0.5

    return 0.0


def _contains_subsequence(sequence: list[str], pattern: tuple[str, ...]) -> bool:
    """Check if pattern appears as a subsequence in sequence."""
    pattern_idx = 0
    for item in sequence:
        if pattern_idx < len(pattern) and item == pattern[pattern_idx]:
            pattern_idx += 1
        if pattern_idx == len(pattern):
            return True
    return False


def annotate_session_rewards(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Annotate a session's messages with reward entries.

    Reads through the message list, computes rewards for each tool result,
    and appends ``role: "reward"`` entries after each tool result.

    Returns:
        New message list with reward entries interleaved.
    """
    annotated: list[dict[str, Any]] = []
    tool_sequence: list[str] = []
    step = 0

    for msg in messages:
        annotated.append(msg)
        role = msg.get("role", "")

        if role == "tool":
            step += 1
            tool_name = msg.get("name", "unknown")
            is_error = msg.get("is_error", False)
            tool_sequence.append(tool_name)

            reward = compute_tool_reward(
                tool_name=tool_name,
                arguments={},
                is_error=is_error,
                step=step,
            )

            reward_entry = {
                "role": "reward",
                "step": reward.step,
                "tool_name": reward.tool_name,
                "reward": reward.reward,
                "components": reward.components,
                "reasoning": reward.reasoning,
            }
            annotated.append(reward_entry)

    # Sequence bonus (applied to the last reward entry if any)
    seq_bonus = compute_sequence_bonus(tool_sequence)
    if seq_bonus > 0 and annotated:
        # Find last reward entry and add sequence bonus
        for i in range(len(annotated) - 1, -1, -1):
            if annotated[i].get("role") == "reward":
                annotated[i] = {
                    **annotated[i],
                    "reward": round(annotated[i]["reward"] + seq_bonus, 2),
                    "components": {
                        **annotated[i]["components"],
                        "efficient_sequence": seq_bonus,
                    },
                    "reasoning": annotated[i]["reasoning"] + "; Efficient tool sequence bonus",
                }
                break

    return annotated


@dataclass(slots=True)
class SessionRewardSummary:
    """Aggregate reward statistics for a session."""

    total_reward: float = 0.0
    step_count: int = 0
    mean_reward: float = 0.0
    successful_steps: int = 0
    failed_steps: int = 0
    components: dict[str, float] = field(default_factory=dict)


def summarize_rewards(
    messages: list[dict[str, Any]],
) -> SessionRewardSummary:
    """Compute aggregate reward statistics from annotated messages."""
    summary = SessionRewardSummary()

    for msg in messages:
        if msg.get("role") != "reward":
            continue
        summary.step_count += 1
        reward = msg.get("reward", 0.0)
        summary.total_reward += reward

        if reward >= 0:
            summary.successful_steps += 1
        else:
            summary.failed_steps += 1

        for key, val in msg.get("components", {}).items():
            summary.components[key] = summary.components.get(key, 0.0) + val

    if summary.step_count > 0:
        summary.mean_reward = round(summary.total_reward / summary.step_count, 3)
    summary.total_reward = round(summary.total_reward, 2)

    return summary
