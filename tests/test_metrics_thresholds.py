"""Tests for active metrics threshold checking and metrics collection."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import patch

import pytest

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


# ---------------------------------------------------------------------------
# Test: LoopMetrics tool call tracking
# ---------------------------------------------------------------------------


class TestLoopMetricsToolCalls:
    def test_record_tool_call_increments_total(self) -> None:
        m = LoopMetrics()
        assert m.tool_calls_total == 0
        m.record_tool_call("bash", 0.5, is_error=False)
        assert m.tool_calls_total == 1
        assert m.tool_errors_total == 0

    def test_record_tool_call_error(self) -> None:
        m = LoopMetrics()
        m.record_tool_call("bash", 1.0, is_error=True)
        assert m.tool_calls_total == 1
        assert m.tool_errors_total == 1

    def test_record_tool_call_updates_histogram(self) -> None:
        m = LoopMetrics()
        m.record_tool_call("file_read", 0.1)
        m.record_tool_call("bash", 0.5)
        assert m._tool_latency_ms.count == 2
        assert m._tool_latency_ms.p50 is not None

    def test_record_multiple_errors(self) -> None:
        m = LoopMetrics()
        for _ in range(3):
            m.record_tool_call("bash", 0.1, is_error=True)
        for _ in range(2):
            m.record_tool_call("file_read", 0.2, is_error=False)
        assert m.tool_calls_total == 5
        assert m.tool_errors_total == 3


# ---------------------------------------------------------------------------
# Test: LoopMetrics tool denials
# ---------------------------------------------------------------------------


class TestLoopMetricsDenials:
    def test_record_tool_denial(self) -> None:
        m = LoopMetrics()
        assert m.tool_denials_total == 0
        m.record_tool_denial()
        assert m.tool_denials_total == 1

    def test_multiple_denials(self) -> None:
        m = LoopMetrics()
        for _ in range(5):
            m.record_tool_denial()
        assert m.tool_denials_total == 5


# ---------------------------------------------------------------------------
# Test: LoopMetrics speculative tracking
# ---------------------------------------------------------------------------


class TestLoopMetricsSpeculative:
    def test_record_speculative_hit(self) -> None:
        m = LoopMetrics()
        m.record_speculative_hit()
        assert m.speculative_hits == 1
        assert m.speculative_misses == 0

    def test_record_speculative_miss(self) -> None:
        m = LoopMetrics()
        m.record_speculative_miss()
        assert m.speculative_misses == 1

    def test_hit_rate_with_no_data(self) -> None:
        m = LoopMetrics()
        assert m.speculative_hit_rate == 0.0

    def test_hit_rate_perfect(self) -> None:
        m = LoopMetrics()
        m.record_speculative_hit()
        assert m.speculative_hit_rate == 1.0

    def test_hit_rate_fifty_percent(self) -> None:
        m = LoopMetrics()
        m.record_speculative_hit()
        m.record_speculative_miss()
        assert m.speculative_hit_rate == 0.5


# ---------------------------------------------------------------------------
# Test: LoopMetrics iteration and LLM call tracking
# ---------------------------------------------------------------------------


class TestLoopMetricsIteration:
    def test_record_iteration(self) -> None:
        m = LoopMetrics()
        m.record_iteration(2.5)
        assert m.iterations == 1
        assert m._loop_duration_ms.count == 1
        assert m._loop_duration_ms.p50 is not None

    def test_record_llm_call(self) -> None:
        m = LoopMetrics()
        m.record_llm_call(1.0)
        m.record_llm_call(2.0)
        assert m._llm_latency_ms.count == 2
        assert m._llm_latency_ms.p50 is not None

    def test_loop_duration_p99_with_many_entries(self) -> None:
        m = LoopMetrics()
        for i in range(100):
            m.record_iteration(float(i) / 10)
        assert m._loop_duration_ms.p99 is not None
        assert m._loop_duration_ms.p50 is not None


# ---------------------------------------------------------------------------
# Test: LoopMetrics token tracking and velocity
# ---------------------------------------------------------------------------


class TestLoopMetricsTokens:
    def test_token_velocity_requires_two_samples(self) -> None:
        m = LoopMetrics()
        assert m.token_velocity_per_min is None

    def test_token_velocity_calculates_correctly(self) -> None:
        m = LoopMetrics()
        now = time.monotonic()
        m._token_samples.append((now - 60, 0))
        m._token_samples.append((now, 1200))
        velocity = m.token_velocity_per_min
        assert velocity is not None
        assert velocity == pytest.approx(1200.0, rel=0.1)

    def test_token_velocity_zero_delta(self) -> None:
        m = LoopMetrics()
        now = time.monotonic()
        m._token_samples.append((now, 100))
        m._token_samples.append((now, 200))
        assert m.token_velocity_per_min is None

    def test_record_token_count(self) -> None:
        m = LoopMetrics()
        m.record_token_count(500)
        m.record_token_count(1000)
        assert len(m._token_samples) == 2

    def test_token_samples_capped_at_60(self) -> None:
        m = LoopMetrics()
        for i in range(65):
            m.record_token_count(i * 10)
        assert len(m._token_samples) == 60


# ---------------------------------------------------------------------------
# Test: LoopMetrics compaction and must_fix
# ---------------------------------------------------------------------------


class TestLoopMetricsOther:
    def test_record_compaction(self) -> None:
        m = LoopMetrics()
        assert m.compactions == 0
        m.record_compaction()
        assert m.compactions == 1

    def test_record_must_fix(self) -> None:
        m = LoopMetrics()
        assert m.must_fix_injections == 0
        m.record_must_fix()
        assert m.must_fix_injections == 1


# ---------------------------------------------------------------------------
# Test: LoopMetrics duration
# ---------------------------------------------------------------------------


class TestLoopMetricsDuration:
    def test_duration_increases(self) -> None:
        m = LoopMetrics()
        d1 = m.duration_seconds
        time.sleep(0.01)
        d2 = m.duration_seconds
        assert d2 > d1


# ---------------------------------------------------------------------------
# Test: LoopMetrics to_dict export
# ---------------------------------------------------------------------------


class TestLoopMetricsToDict:
    def test_to_dict_empty_metrics(self) -> None:
        m = LoopMetrics()
        d = m.to_dict()
        assert d["iterations"] == 0
        assert d["tool_calls_total"] == 0
        assert d["tool_errors_total"] == 0
        assert d["tool_denials_total"] == 0
        assert d["speculative_hits"] == 0
        assert d["speculative_misses"] == 0
        assert d["speculative_hit_rate"] == 0.0
        assert d["must_fix_injections"] == 0
        assert d["compactions"] == 0
        assert d["token_velocity_per_min"] is None

    def test_to_dict_with_data(self) -> None:
        m = LoopMetrics()
        m.record_tool_call("bash", 0.5)
        m.record_tool_call("bash", 0.3, is_error=True)
        m.record_llm_call(1.0)
        m.record_iteration(2.0)
        m.record_speculative_hit()
        m.record_compaction()
        m.record_must_fix()

        d = m.to_dict()
        assert d["tool_calls_total"] == 2
        assert d["tool_errors_total"] == 1
        assert d["speculative_hits"] == 1
        assert d["compactions"] == 1
        assert d["must_fix_injections"] == 1
        assert d["loop_duration_ms"]["count"] == 1
        assert d["tool_latency_ms"]["count"] == 2
        assert d["llm_latency_ms"]["count"] == 1

    def test_to_dict_with_histogram_stats(self) -> None:
        m = LoopMetrics()
        for i in range(10):
            m.record_tool_call("bash", float(i + 1) * 0.1)

        d = m.to_dict()
        assert d["tool_latency_ms"]["count"] == 10
        assert d["tool_latency_ms"]["p50"] is not None
        assert d["tool_latency_ms"]["p99"] is not None
        assert d["tool_latency_ms"]["mean"] is not None


# ---------------------------------------------------------------------------
# Test: _Histogram internal
# ---------------------------------------------------------------------------


class TestHistogram:
    def test_empty_histogram_properties(self) -> None:
        from godspeed.observability.metrics import _Histogram
        h = _Histogram()
        assert h.count == 0
        assert h.p50 is None
        assert h.p99 is None
        assert h.mean is None

    def test_single_value_histogram(self) -> None:
        from godspeed.observability.metrics import _Histogram
        h = _Histogram()
        h.observe(42.0)
        assert h.count == 1
        assert h.p50 == 42.0
        assert h.p99 == 42.0
        assert h.mean == 42.0

    def test_histogram_retains_last_100(self) -> None:
        from godspeed.observability.metrics import _Histogram
        h = _Histogram()
        for i in range(150):
            h.observe(float(i))
        assert h.count == 100
        assert h.p50 >= 49.0

    def test_histogram_p99(self) -> None:
        from godspeed.observability.metrics import _Histogram
        h = _Histogram()
        for i in range(100):
            h.observe(float(i))
        assert h.p99 >= 98.0


# ---------------------------------------------------------------------------
# Test: MetricsSink
# ---------------------------------------------------------------------------


class TestMetricsSink:
    def test_emit_to_stdout(self) -> None:
        from godspeed.observability.metrics import MetricsSink
        sink = MetricsSink(path=None)
        sink.emit("test_event", {"value": 42})
        sink.close()

    def test_emit_to_file(self, tmp_path: Path) -> None:
        from godspeed.observability.metrics import MetricsSink
        import json

        sink = MetricsSink(path=tmp_path / "metrics" / "test.jsonl")
        sink.emit("tool_call", {"tool": "bash", "duration": 0.5})
        sink.emit("llm_call", {"tokens": 100})
        sink.close()

        data = (tmp_path / "metrics" / "test.jsonl").read_text()
        lines = data.strip().split("\n")
        assert len(lines) == 2
        record = json.loads(lines[0])
        assert record["event"] == "tool_call"
        assert record["tool"] == "bash"

    def test_emit_to_file_creates_parent_dirs(self, tmp_path: Path) -> None:
        from godspeed.observability.metrics import MetricsSink
        sink = MetricsSink(path=tmp_path / "deep" / "nested" / "metrics.jsonl")
        sink.emit("test", {"key": "val"})
        sink.close()
        assert (tmp_path / "deep" / "nested" / "metrics.jsonl").exists()

    def test_emit_write_failure_graceful(self, tmp_path: Path) -> None:
        from godspeed.observability.metrics import MetricsSink
        sink = MetricsSink(path=tmp_path / "metrics.jsonl")
        with patch("builtins.open", side_effect=OSError("disk full")):
            sink.emit("test", {"key": "val"})
        sink.close()

    def test_close_when_no_file(self) -> None:
        from godspeed.observability.metrics import MetricsSink
        sink = MetricsSink(path=None)
        sink.close()

    def test_close_suppresses_os_error(self, tmp_path: Path) -> None:
        from godspeed.observability.metrics import MetricsSink
        sink = MetricsSink(path=tmp_path / "metrics.jsonl")
        sink.emit("test", {"key": "val"})
        with patch.object(sink._file, "close", side_effect=OSError("close error")):
            sink.close()
        assert sink._file is None

    def test_context_manager(self, tmp_path: Path) -> None:
        from godspeed.observability.metrics import MetricsSink
        with MetricsSink(path=tmp_path / "ctx_metrics.jsonl") as sink:
            sink.emit("test", {"ctx": True})
        assert (tmp_path / "ctx_metrics.jsonl").exists()

    def test_context_manager_exit(self, tmp_path: Path) -> None:
        from godspeed.observability.metrics import MetricsSink
        sink = MetricsSink(path=tmp_path / "exit_metrics.jsonl")
        sink.__enter__()
        sink.__exit__(None, None, None)
        assert sink._file is None

    def test_emit_without_path_logs_debug(self) -> None:
        from godspeed.observability.metrics import MetricsSink
        sink = MetricsSink(path=None)
        sink.emit("test", {"key": "value"})
        sink.close()

    def test_emit_multiple_events_to_same_file(self, tmp_path: Path) -> None:
        from godspeed.observability.metrics import MetricsSink
        sink = MetricsSink(path=tmp_path / "multi.jsonl")
        for i in range(10):
            sink.emit("iteration", {"i": i})
        sink.close()

        lines = (tmp_path / "multi.jsonl").read_text().strip().split("\n")
        assert len(lines) == 10


# ---------------------------------------------------------------------------
# Test: LoopMetrics reset (via creating new instance)
# ---------------------------------------------------------------------------


class TestLoopMetricsReset:
    def test_new_instance_starts_fresh(self) -> None:
        m1 = LoopMetrics()
        m1.record_tool_call("bash", 1.0)
        m1.record_iteration(0.5)
        assert m1.tool_calls_total == 1
        assert m1.iterations == 1

        m2 = LoopMetrics()
        assert m2.tool_calls_total == 0
        assert m2.iterations == 0

    def test_to_dict_after_reset_like_behavior(self) -> None:
        m = LoopMetrics()
        m.record_tool_call("bash", 1.0)
        m.record_iteration(0.5)

        d = m.to_dict()
        assert d["tool_calls_total"] == 1
        assert d["iterations"] == 1


# ---------------------------------------------------------------------------
# Test: Alert dataclass
# ---------------------------------------------------------------------------


class TestAlertDataClass:
    def test_alert_creation(self) -> None:
        from godspeed.observability.metrics import Alert, AlertSeverity
        a = Alert(
            severity=AlertSeverity.WARNING,
            metric="test_metric",
            value=42.0,
            threshold=10.0,
            message="test alert",
        )
        assert a.severity == AlertSeverity.WARNING
        assert a.metric == "test_metric"
        assert a.value == 42.0

    def test_alert_critical(self) -> None:
        from godspeed.observability.metrics import Alert, AlertSeverity
        a = Alert(
            severity=AlertSeverity.CRITICAL,
            metric="error_rate",
            value=0.5,
            threshold=0.25,
            message="critical error rate",
        )
        assert a.severity == AlertSeverity.CRITICAL


# ---------------------------------------------------------------------------
# Test: MetricsThresholds dataclass
# ---------------------------------------------------------------------------


class TestMetricsThresholdsDataClass:
    def test_default_values(self) -> None:
        t = MetricsThresholds()
        assert t.token_velocity_warning == 10_000.0
        assert t.token_velocity_critical == 50_000.0
        assert t.error_rate_warning == 0.10
        assert t.error_rate_critical == 0.25
        assert t.llm_latency_p99_warning_ms == 30_000.0
        assert t.llm_latency_p99_critical_ms == 60_000.0
        assert t.loop_duration_p99_warning_ms == 5_000.0
        assert t.loop_duration_p99_critical_ms == 15_000.0
        assert t.speculative_hit_rate_warning == 0.30

    def test_all_disabled_thresholds(self) -> None:
        t = MetricsThresholds(
            token_velocity_warning=None,
            token_velocity_critical=None,
            error_rate_warning=None,
            error_rate_critical=None,
            llm_latency_p99_warning_ms=None,
            llm_latency_p99_critical_ms=None,
            loop_duration_p99_warning_ms=None,
            loop_duration_p99_critical_ms=None,
            speculative_hit_rate_warning=None,
        )
        m = LoopMetrics()
        for _ in range(100):
            m.record_llm_call(100.0)
        alerts = check_thresholds(m, t)
        assert alerts == []


# ---------------------------------------------------------------------------
# Test: AlertSeverity enum
# ---------------------------------------------------------------------------


class TestAlertSeverity:
    def test_warning_value(self) -> None:
        assert AlertSeverity.WARNING == "warning"

    def test_critical_value(self) -> None:
        assert AlertSeverity.CRITICAL == "critical"

    def test_str_equality(self) -> None:
        assert str(AlertSeverity.WARNING) == "warning"
