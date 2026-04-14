"""Tests for per-step reward annotations."""

from __future__ import annotations

from godspeed.training.rewards import (
    SessionRewardSummary,
    StepReward,
    _contains_subsequence,
    annotate_session_rewards,
    compute_sequence_bonus,
    compute_tool_reward,
    summarize_rewards,
)


class TestComputeToolReward:
    def test_successful_execution(self) -> None:
        r = compute_tool_reward("file_read", {}, is_error=False, step=1)
        assert r.reward == 1.0
        assert r.components["execution_success"] == 1.0
        assert r.tool_name == "file_read"
        assert r.step == 1

    def test_failed_execution(self) -> None:
        r = compute_tool_reward("file_edit", {}, is_error=True, step=2)
        assert r.reward == -0.5
        assert "execution_failed" in r.components

    def test_permission_denied(self) -> None:
        r = compute_tool_reward("shell", {}, is_error=False, permission_denied=True)
        assert r.reward == -0.5
        assert "permission_denied" in r.components

    def test_dangerous_command(self) -> None:
        r = compute_tool_reward("shell", {}, is_error=False, is_dangerous=True)
        # success (1.0) + dangerous (-1.0) = 0.0
        assert r.reward == 0.0
        assert r.components["dangerous_command"] == -1.0

    def test_verify_passed_first_try(self) -> None:
        r = compute_tool_reward("file_edit", {}, is_error=False, verify_passed=True)
        assert r.reward == 1.5
        assert r.components["verify_passed"] == 0.5

    def test_verify_passed_after_retry(self) -> None:
        r = compute_tool_reward(
            "file_edit", {}, is_error=False, verify_passed=True, was_retried=True
        )
        assert r.reward == 1.25
        assert r.components["verify_retry_fixed"] == 0.25

    def test_verify_failed(self) -> None:
        r = compute_tool_reward("file_edit", {}, is_error=False, verify_passed=False)
        assert r.reward == 0.75
        assert r.components["verify_failed"] == -0.25

    def test_combined_penalties(self) -> None:
        r = compute_tool_reward("shell", {}, is_error=True, is_dangerous=True, step=3)
        # failed (-0.5) + dangerous (-1.0) = -1.5
        assert r.reward == -1.5
        assert r.step == 3

    def test_reasoning_populated(self) -> None:
        r = compute_tool_reward("file_read", {}, is_error=False)
        assert "successfully" in r.reasoning

    def test_returns_step_reward_type(self) -> None:
        r = compute_tool_reward("file_read", {}, is_error=False)
        assert isinstance(r, StepReward)


class TestSequenceBonus:
    def test_grep_read_edit(self) -> None:
        seq = ["grep_search", "file_read", "file_edit"]
        assert compute_sequence_bonus(seq) == 0.5

    def test_glob_read_edit(self) -> None:
        seq = ["glob_search", "file_read", "file_edit"]
        assert compute_sequence_bonus(seq) == 0.5

    def test_read_edit_short(self) -> None:
        seq = ["file_read", "file_edit"]
        assert compute_sequence_bonus(seq) == 0.5

    def test_web_search_fetch(self) -> None:
        seq = ["web_search", "web_fetch"]
        assert compute_sequence_bonus(seq) == 0.5

    def test_no_pattern(self) -> None:
        seq = ["shell", "git"]
        assert compute_sequence_bonus(seq) == 0.0

    def test_empty_sequence(self) -> None:
        assert compute_sequence_bonus([]) == 0.0

    def test_single_tool(self) -> None:
        assert compute_sequence_bonus(["file_read"]) == 0.0

    def test_pattern_with_noise(self) -> None:
        # Pattern with extra tools interspersed
        seq = ["grep_search", "repo_map", "file_read", "verify", "file_edit"]
        assert compute_sequence_bonus(seq) == 0.5

    def test_contains_subsequence(self) -> None:
        assert _contains_subsequence(["a", "b", "c", "d"], ("a", "c", "d")) is True

    def test_not_contains_subsequence(self) -> None:
        assert _contains_subsequence(["a", "b", "c"], ("c", "a")) is False


class TestAnnotateSessionRewards:
    def test_adds_reward_entries(self) -> None:
        messages = [
            {"role": "user", "content": "fix bug"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "c1", "name": "file_read"}],
            },
            {
                "role": "tool",
                "tool_call_id": "c1",
                "name": "file_read",
                "content": "...",
                "is_error": False,
            },
            {"role": "assistant", "content": "Done"},
        ]
        annotated = annotate_session_rewards(messages)
        reward_entries = [m for m in annotated if m.get("role") == "reward"]
        assert len(reward_entries) == 1
        assert reward_entries[0]["reward"] == 1.0

    def test_error_step_negative_reward(self) -> None:
        messages = [
            {
                "role": "tool",
                "name": "shell",
                "content": "fail",
                "is_error": True,
            },
        ]
        annotated = annotate_session_rewards(messages)
        reward_entries = [m for m in annotated if m.get("role") == "reward"]
        assert len(reward_entries) == 1
        assert reward_entries[0]["reward"] < 0

    def test_sequence_bonus_applied(self) -> None:
        messages = [
            {
                "role": "tool",
                "name": "grep_search",
                "content": "found",
                "is_error": False,
            },
            {
                "role": "tool",
                "name": "file_read",
                "content": "content",
                "is_error": False,
            },
            {
                "role": "tool",
                "name": "file_edit",
                "content": "edited",
                "is_error": False,
            },
        ]
        annotated = annotate_session_rewards(messages)
        reward_entries = [m for m in annotated if m.get("role") == "reward"]
        assert len(reward_entries) == 3
        # Last entry should have sequence bonus
        last_reward = reward_entries[-1]
        assert "efficient_sequence" in last_reward["components"]
        assert last_reward["reward"] == 1.5  # 1.0 + 0.5

    def test_preserves_original_messages(self) -> None:
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        annotated = annotate_session_rewards(messages)
        # No tool calls → no rewards added
        assert len(annotated) == 2
        assert annotated == messages

    def test_step_numbers_sequential(self) -> None:
        messages = [
            {"role": "tool", "name": "a", "content": "x", "is_error": False},
            {"role": "tool", "name": "b", "content": "y", "is_error": False},
            {"role": "tool", "name": "c", "content": "z", "is_error": False},
        ]
        annotated = annotate_session_rewards(messages)
        reward_entries = [m for m in annotated if m.get("role") == "reward"]
        steps = [r["step"] for r in reward_entries]
        assert steps == [1, 2, 3]


class TestSummarizeRewards:
    def test_basic_summary(self) -> None:
        messages = [
            {"role": "reward", "reward": 1.0, "components": {"success": 1.0}},
            {"role": "reward", "reward": 1.5, "components": {"success": 1.0, "verify": 0.5}},
            {"role": "reward", "reward": -0.5, "components": {"failed": -0.5}},
        ]
        summary = summarize_rewards(messages)
        assert summary.step_count == 3
        assert summary.total_reward == 2.0
        assert summary.successful_steps == 2
        assert summary.failed_steps == 1
        assert summary.mean_reward == round(2.0 / 3, 3)

    def test_empty_messages(self) -> None:
        summary = summarize_rewards([])
        assert summary.step_count == 0
        assert summary.total_reward == 0.0
        assert summary.mean_reward == 0.0

    def test_ignores_non_reward(self) -> None:
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "reward", "reward": 1.0, "components": {"success": 1.0}},
        ]
        summary = summarize_rewards(messages)
        assert summary.step_count == 1

    def test_aggregates_components(self) -> None:
        messages = [
            {"role": "reward", "reward": 1.0, "components": {"success": 1.0}},
            {
                "role": "reward",
                "reward": 1.5,
                "components": {"success": 1.0, "verify": 0.5},
            },
        ]
        summary = summarize_rewards(messages)
        assert summary.components["success"] == 2.0
        assert summary.components["verify"] == 0.5

    def test_returns_summary_type(self) -> None:
        summary = summarize_rewards([])
        assert isinstance(summary, SessionRewardSummary)
