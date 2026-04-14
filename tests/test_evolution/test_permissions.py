"""Tests for permission pattern learning."""

from __future__ import annotations

import pytest

from godspeed.evolution.permissions import PermissionAdvisor, PermissionSuggestion
from godspeed.evolution.trace_analyzer import SessionTrace

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _session(
    denials: tuple[tuple[str, str], ...] = (),
    grants: tuple[str, ...] = (),
) -> SessionTrace:
    return SessionTrace(
        session_id="s1",
        tool_calls=(),
        errors=(),
        permission_denials=denials,
        permission_grants=grants,
        total_latency_ms=0.0,
        model="",
    )


# ---------------------------------------------------------------------------
# Test: analyze_denials
# ---------------------------------------------------------------------------


class TestAnalyzeDenials:
    def test_frequent_denials_flagged(self) -> None:
        sessions = [
            _session(denials=tuple(("bash", "not allowed") for _ in range(6))),
        ]
        advisor = PermissionAdvisor(denial_threshold=5)
        suggestions = advisor.analyze_denials(sessions)

        assert len(suggestions) == 1
        assert suggestions[0].tool_name == "bash"
        assert suggestions[0].action == "add_to_allowlist"
        assert suggestions[0].denial_count == 6

    def test_below_threshold_excluded(self) -> None:
        sessions = [
            _session(denials=(("bash", "not allowed"), ("bash", "not allowed"))),
        ]
        advisor = PermissionAdvisor(denial_threshold=5)
        suggestions = advisor.analyze_denials(sessions)
        assert suggestions == []

    def test_rationale_includes_reasons(self) -> None:
        sessions = [
            _session(denials=tuple(("bash", "blocked by policy") for _ in range(5))),
        ]
        advisor = PermissionAdvisor(denial_threshold=5)
        suggestions = advisor.analyze_denials(sessions)
        assert "blocked by policy" in suggestions[0].rationale


# ---------------------------------------------------------------------------
# Test: analyze_approvals
# ---------------------------------------------------------------------------


class TestAnalyzeApprovals:
    def test_frequent_grants_suggest_pre_approve(self) -> None:
        sessions = [
            _session(grants=tuple("file_read" for _ in range(6))),
        ]
        advisor = PermissionAdvisor(grant_threshold=5)
        suggestions = advisor.analyze_approvals(sessions)

        assert len(suggestions) == 1
        assert suggestions[0].tool_name == "file_read"
        assert suggestions[0].action == "pre_approve"

    def test_mixed_grant_denial_excluded(self) -> None:
        sessions = [
            _session(
                grants=tuple("bash" for _ in range(6)),
                denials=(("bash", "denied once"),),
            ),
        ]
        advisor = PermissionAdvisor(grant_threshold=5)
        suggestions = advisor.analyze_approvals(sessions)
        assert suggestions == []

    def test_below_threshold_excluded(self) -> None:
        sessions = [_session(grants=("file_read",))]
        advisor = PermissionAdvisor(grant_threshold=5)
        assert advisor.analyze_approvals(sessions) == []


# ---------------------------------------------------------------------------
# Test: generate_permission_config
# ---------------------------------------------------------------------------


class TestGeneratePermissionConfig:
    def test_generates_config(self) -> None:
        suggestions = [
            PermissionSuggestion(
                tool_name="file_read",
                action="pre_approve",
                denial_count=0,
                grant_count=10,
                rationale="safe",
            ),
            PermissionSuggestion(
                tool_name="bash",
                action="add_to_allowlist",
                denial_count=8,
                grant_count=0,
                rationale="frequent",
            ),
        ]
        advisor = PermissionAdvisor()
        config = advisor.generate_permission_config(suggestions)

        assert "permissions" in config
        assert "bash" in config["permissions"]["allow"]
        assert "file_read" in config["permissions"]["allow"]

    def test_empty_suggestions(self) -> None:
        advisor = PermissionAdvisor()
        config = advisor.generate_permission_config([])
        assert config["permissions"]["allow"] == []


# ---------------------------------------------------------------------------
# Test: get_all_suggestions
# ---------------------------------------------------------------------------


class TestGetAllSuggestions:
    def test_combines_denials_and_approvals(self) -> None:
        sessions = [
            _session(
                denials=tuple(("bash", "blocked") for _ in range(6)),
                grants=tuple("file_read" for _ in range(6)),
            ),
        ]
        advisor = PermissionAdvisor(denial_threshold=5, grant_threshold=5)
        all_suggestions = advisor.get_all_suggestions(sessions)

        tools = {s.tool_name for s in all_suggestions}
        assert "bash" in tools
        assert "file_read" in tools


# ---------------------------------------------------------------------------
# Test: data structures
# ---------------------------------------------------------------------------


class TestPermissionSuggestion:
    def test_frozen(self) -> None:
        s = PermissionSuggestion(
            tool_name="bash",
            action="add_to_allowlist",
            denial_count=5,
            grant_count=0,
            rationale="test",
        )
        with pytest.raises(AttributeError):
            s.action = "changed"  # type: ignore[misc]
