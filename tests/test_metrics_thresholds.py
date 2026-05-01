"""Tests for active metrics threshold checking."""

from __future__ import annotations

import time

from godspeed.observability.metrics import (
    AlertSeverity,
    LoopMetrics,
    MetricsThresholds,
    check_thresholds,
)


class TestCheckThresholds:
    def test_no_alerts_when_healthy(self) -> None:
        m = LoopMetrics()
        m.record_tool_call("x", 0.1, is_error=False)
        m.record_llm_call(0.5)
        m.record_iteration(0.2)
        alerts = check_thresholds(m)
        assert alerts == []

    def test_token_velocity_warning(self) -> None:
        m = LoopMetrics()
        # Simulate 6k tokens in 30s -> 12k/min (above 10k warning)
        now = time.monotonic()
        m._token_samples.append((now - 30, 0))
        m._token_samples.append((now, 6000))
        alerts = check_thresholds(m)
        assert len(alerts) == 1
        assert alerts[0].severity == AlertSeverity.WARNING
        assert alerts[0].metric == "token_velocity_per_min"

    def test_token_velocity_critical(self) -> None:
        m = LoopMetrics()
        now = time.monotonic()
        m._token_samples.append((now - 30, 0))
        m._token_samples.append((now, 30_000))
        alerts = check_thresholds(m)
        assert len(alerts) == 1
        assert alerts[0].severity == AlertSeverity.CRITICAL
        assert alerts[0].metric == "token_velocity_per_min"

    def test_error_rate_warning(self) -> None:
        m = LoopMetrics()
        for _ in range(8):
            m.record_tool_call("x", 0.1, is_error=False)
        for _ in range(2):
            m.record_tool_call("x", 0.1, is_error=True)
        # 20% error rate > 10% warning
        alerts = check_thresholds(m)
        assert any(a.metric == "error_rate" and a.severity == AlertSeverity.WARNING for a in alerts)

    def test_error_rate_critical(self) -> None:
        m = LoopMetrics()
        for _ in range(3):
            m.record_tool_call("x", 0.1, is_error=False)
        for _ in range(7):
            m.record_tool_call("x", 0.1, is_error=True)
        # 70% error rate > 25% critical
        alerts = check_thresholds(m)
        assert any(
            a.metric == "error_rate" and a.severity == AlertSeverity.CRITICAL for a in alerts
        )

    def test_llm_latency_p99_warning(self) -> None:
        m = LoopMetrics()
        for _ in range(100):
            m.record_llm_call(35.0)  # 35s = 35,000ms
        alerts = check_thresholds(m)
        assert any(
            a.metric == "llm_latency_p99_ms" and a.severity == AlertSeverity.WARNING for a in alerts
        )

    def test_llm_latency_p99_critical(self) -> None:
        m = LoopMetrics()
        for _ in range(100):
            m.record_llm_call(70.0)  # 70s = 70,000ms
        alerts = check_thresholds(m)
        assert any(
            a.metric == "llm_latency_p99_ms" and a.severity == AlertSeverity.CRITICAL
            for a in alerts
        )

    def test_loop_duration_p99_warning(self) -> None:
        m = LoopMetrics()
        for _ in range(100):
            m.record_iteration(6.0)  # 6s = 6,000ms
        alerts = check_thresholds(m)
        assert any(
            a.metric == "loop_duration_p99_ms" and a.severity == AlertSeverity.WARNING
            for a in alerts
        )

    def test_loop_duration_p99_critical(self) -> None:
        m = LoopMetrics()
        for _ in range(100):
            m.record_iteration(20.0)  # 20s = 20,000ms
        alerts = check_thresholds(m)
        assert any(
            a.metric == "loop_duration_p99_ms" and a.severity == AlertSeverity.CRITICAL
            for a in alerts
        )

    def test_speculative_hit_rate_warning(self) -> None:
        m = LoopMetrics()
        m.record_speculative_hit()
        m.record_speculative_miss()
        m.record_speculative_miss()
        # 33% hit rate > 30% warning? No — 33% is above 30% so no warning.
        # Let's make it 20%.
        m2 = LoopMetrics()
        m2.record_speculative_hit()
        for _ in range(4):
            m2.record_speculative_miss()
        alerts = check_thresholds(m2)
        assert any(
            a.metric == "speculative_hit_rate" and a.severity == AlertSeverity.WARNING
            for a in alerts
        )

    def test_no_alert_when_hit_rate_ok(self) -> None:
        m = LoopMetrics()
        m.record_speculative_hit()
        m.record_speculative_hit()
        m.record_speculative_miss()
        # 67% hit rate — no warning
        alerts = check_thresholds(m)
        assert not any(a.metric == "speculative_hit_rate" for a in alerts)

    def test_disabled_threshold(self) -> None:
        m = LoopMetrics()
        for _ in range(100):
            m.record_llm_call(70.0)
        thresholds = MetricsThresholds(
            llm_latency_p99_warning_ms=None,
            llm_latency_p99_critical_ms=None,
        )
        alerts = check_thresholds(m, thresholds)
        assert not any(a.metric == "llm_latency_p99_ms" for a in alerts)

    def test_critical_sorted_before_warning(self) -> None:
        m = LoopMetrics()
        for _ in range(100):
            m.record_llm_call(70.0)
        m.record_tool_call("x", 0.1, is_error=True)
        m.record_tool_call("x", 0.1, is_error=False)
        alerts = check_thresholds(m)
        severities = [a.severity for a in alerts]
        assert severities[0] == AlertSeverity.CRITICAL

    def test_alert_message_contains_value_and_threshold(self) -> None:
        m = LoopMetrics()
        for _ in range(100):
            m.record_llm_call(70.0)
        alerts = check_thresholds(m)
        assert alerts
        assert "70,000ms" in alerts[0].message
        assert "critical threshold" in alerts[0].message

    def test_no_alerts_on_empty_metrics(self) -> None:
        m = LoopMetrics()
        alerts = check_thresholds(m)
        assert alerts == []
