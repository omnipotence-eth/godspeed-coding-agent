"""Tests for cross-session learning and regression detection."""

from __future__ import annotations

import pytest

from godspeed.evolution.cross_session import (
    AggregateReport,
    CrossSessionAnalyzer,
    RegressionAlert,
)
from godspeed.evolution.trace_analyzer import SessionTrace, ToolCall

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _tc(tool: str = "file_read", is_error: bool = False, latency: float = 50.0) -> ToolCall:
    return ToolCall(
        tool_name=tool,
        arguments={},
        output_length=100,
        is_error=is_error,
        latency_ms=latency,
        outcome="error" if is_error else "success",
    )


def _session(
    session_id: str = "s1",
    model: str = "ollama/gemma3:12b",
    tool_calls: tuple[ToolCall, ...] = (),
    latency: float = 100.0,
) -> SessionTrace:
    errors = tuple(tc for tc in tool_calls if tc.is_error)
    return SessionTrace(
        session_id=session_id,
        tool_calls=tool_calls,
        errors=errors,
        permission_denials=(),
        permission_grants=(),
        total_latency_ms=latency,
        model=model,
    )


# ---------------------------------------------------------------------------
# Test: aggregate_sessions
# ---------------------------------------------------------------------------


class TestAggregateSessions:
    def test_basic_aggregation(self) -> None:
        sessions = [
            _session("s1", tool_calls=(_tc(), _tc(is_error=True))),
            _session("s2", tool_calls=(_tc(), _tc())),
        ]
        analyzer = CrossSessionAnalyzer()
        report = analyzer.aggregate_sessions(sessions)

        assert report.total_sessions == 2
        assert report.total_tool_calls == 4
        assert report.total_errors == 1
        assert report.error_rate == 0.25

    def test_empty_sessions(self) -> None:
        analyzer = CrossSessionAnalyzer()
        report = analyzer.aggregate_sessions([])
        assert report.total_sessions == 0
        assert report.error_rate == 0.0

    def test_model_distribution(self) -> None:
        sessions = [
            _session("s1", model="claude"),
            _session("s2", model="claude"),
            _session("s3", model="ollama/gemma3"),
        ]
        analyzer = CrossSessionAnalyzer()
        report = analyzer.aggregate_sessions(sessions)
        models = dict(report.model_distribution)
        assert models["claude"] == 2
        assert models["ollama/gemma3"] == 1


# ---------------------------------------------------------------------------
# Test: model_specific_analysis
# ---------------------------------------------------------------------------


class TestModelSpecificAnalysis:
    def test_filters_by_model(self) -> None:
        sessions = [
            _session("s1", model="claude", tool_calls=(_tc(is_error=True),)),
            _session("s2", model="ollama", tool_calls=(_tc(),)),
            _session("s3", model="claude", tool_calls=(_tc(),)),
        ]
        analyzer = CrossSessionAnalyzer()
        report = analyzer.model_specific_analysis(sessions, "claude")

        assert report.model == "claude"
        assert report.sessions == 2

    def test_no_matching_sessions(self) -> None:
        sessions = [_session("s1", model="claude")]
        analyzer = CrossSessionAnalyzer()
        report = analyzer.model_specific_analysis(sessions, "gpt-4")
        assert report.sessions == 0


# ---------------------------------------------------------------------------
# Test: track_evolution_impact
# ---------------------------------------------------------------------------


class TestTrackEvolutionImpact:
    def test_improvement_detected(self) -> None:
        before = [_session("s1", tool_calls=(_tc("bash", is_error=True), _tc("bash")))]
        after = [_session("s2", tool_calls=(_tc("bash"), _tc("bash")))]

        analyzer = CrossSessionAnalyzer()
        impact = analyzer.track_evolution_impact(before, after, "bash")

        assert impact.before_error_rate == 0.5
        assert impact.after_error_rate == 0.0
        assert impact.delta < 0
        assert impact.improved is True

    def test_regression_detected(self) -> None:
        before = [_session("s1", tool_calls=(_tc("bash"),))]
        after = [_session("s2", tool_calls=(_tc("bash", is_error=True),))]

        analyzer = CrossSessionAnalyzer()
        impact = analyzer.track_evolution_impact(before, after, "bash")

        assert impact.delta > 0
        assert impact.improved is False


# ---------------------------------------------------------------------------
# Test: detect_regression
# ---------------------------------------------------------------------------


class TestDetectRegression:
    def test_regression_flagged(self) -> None:
        before = [_session("s1", tool_calls=(_tc("bash"),) * 10)]
        after = [
            _session(
                "s2",
                tool_calls=tuple(_tc("bash", is_error=(i < 5)) for i in range(10)),
            )
        ]

        analyzer = CrossSessionAnalyzer()
        alerts = analyzer.detect_regression(before, after, threshold=0.1)

        assert len(alerts) == 1
        assert alerts[0].artifact_id == "bash"
        assert alerts[0].delta > 0

    def test_no_regression(self) -> None:
        before = [_session("s1", tool_calls=(_tc("bash"),))]
        after = [_session("s2", tool_calls=(_tc("bash"),))]

        analyzer = CrossSessionAnalyzer()
        alerts = analyzer.detect_regression(before, after, threshold=0.1)
        assert alerts == []

    def test_recommendation_severity(self) -> None:
        # Create severe regression (100% error rate vs 0%)
        before = [_session("s1", tool_calls=(_tc("bash"),) * 10)]
        after = [_session("s2", tool_calls=(_tc("bash", is_error=True),) * 10)]

        analyzer = CrossSessionAnalyzer()
        alerts = analyzer.detect_regression(before, after, threshold=0.1)

        assert alerts[0].recommendation == "rollback"


# ---------------------------------------------------------------------------
# Test: data structures frozen
# ---------------------------------------------------------------------------


class TestDataStructures:
    def test_aggregate_report_frozen(self) -> None:
        report = AggregateReport(
            total_sessions=0,
            total_tool_calls=0,
            total_errors=0,
            error_rate=0.0,
            tool_error_rates=(),
            model_distribution=(),
            avg_session_latency_ms=0.0,
        )
        with pytest.raises(AttributeError):
            report.total_sessions = 5  # type: ignore[misc]

    def test_regression_alert_frozen(self) -> None:
        alert = RegressionAlert(
            artifact_id="bash",
            record_id="",
            before_error_rate=0.0,
            after_error_rate=0.5,
            delta=0.5,
            recommendation="rollback",
        )
        with pytest.raises(AttributeError):
            alert.recommendation = "monitor"  # type: ignore[misc]
