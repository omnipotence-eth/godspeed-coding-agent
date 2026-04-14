"""Cross-session learning — aggregate insights and detect regressions.

Analyzes patterns across multiple sessions, provides model-specific tuning
recommendations, and detects regressions after mutations are applied.
"""

from __future__ import annotations

import dataclasses
import logging
from collections import Counter

from godspeed.evolution.trace_analyzer import (
    SessionTrace,
    TraceAnalyzer,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class AggregateReport:
    """Combined insights from multiple sessions."""

    total_sessions: int
    total_tool_calls: int
    total_errors: int
    error_rate: float
    tool_error_rates: tuple[tuple[str, float, int], ...]  # (tool_name, error_rate, count)
    model_distribution: tuple[tuple[str, int], ...]  # (model_name, session_count)
    avg_session_latency_ms: float


@dataclasses.dataclass(frozen=True, slots=True)
class ModelReport:
    """Model-specific analysis."""

    model: str
    sessions: int
    tool_error_rates: tuple[tuple[str, float, int], ...]
    avg_latency_ms: float
    most_failed_tool: str
    most_failed_rate: float


@dataclasses.dataclass(frozen=True, slots=True)
class ImpactReport:
    """Before/after comparison for an evolution change."""

    artifact_id: str
    before_error_rate: float
    after_error_rate: float
    delta: float  # negative = improvement
    before_sessions: int
    after_sessions: int
    improved: bool


@dataclasses.dataclass(frozen=True, slots=True)
class RegressionAlert:
    """Alert for a detected regression after mutation."""

    artifact_id: str
    record_id: str
    before_error_rate: float
    after_error_rate: float
    delta: float
    recommendation: str  # "rollback" | "monitor" | "investigate"


# ---------------------------------------------------------------------------
# Cross-Session Analyzer
# ---------------------------------------------------------------------------


class CrossSessionAnalyzer:
    """Aggregate insights across sessions and detect regressions."""

    def __init__(self) -> None:
        self._analyzer = TraceAnalyzer()

    def aggregate_sessions(self, sessions: list[SessionTrace]) -> AggregateReport:
        """Combine insights from multiple sessions."""
        total_calls = 0
        total_errors = 0
        tool_calls: Counter[str] = Counter()
        tool_errors: Counter[str] = Counter()
        model_counts: Counter[str] = Counter()
        total_latency = 0.0

        for session in sessions:
            model_counts[session.model or "unknown"] += 1
            total_latency += session.total_latency_ms
            for tc in session.tool_calls:
                total_calls += 1
                tool_calls[tc.tool_name] += 1
                if tc.is_error:
                    total_errors += 1
                    tool_errors[tc.tool_name] += 1

        error_rate = total_errors / total_calls if total_calls > 0 else 0.0

        tool_error_rates: list[tuple[str, float, int]] = []
        for tool_name, count in tool_calls.most_common():
            errs = tool_errors.get(tool_name, 0)
            rate = errs / count if count > 0 else 0.0
            tool_error_rates.append((tool_name, rate, count))

        # Sort by error rate descending
        tool_error_rates.sort(key=lambda x: x[1], reverse=True)

        avg_latency = total_latency / len(sessions) if sessions else 0.0

        return AggregateReport(
            total_sessions=len(sessions),
            total_tool_calls=total_calls,
            total_errors=total_errors,
            error_rate=error_rate,
            tool_error_rates=tuple(tool_error_rates),
            model_distribution=tuple(model_counts.most_common()),
            avg_session_latency_ms=avg_latency,
        )

    def model_specific_analysis(self, sessions: list[SessionTrace], model: str) -> ModelReport:
        """Analyze sessions for a specific model."""
        model_sessions = [s for s in sessions if s.model == model]

        tool_calls: Counter[str] = Counter()
        tool_errors: Counter[str] = Counter()
        total_latency = 0.0

        for session in model_sessions:
            total_latency += session.total_latency_ms
            for tc in session.tool_calls:
                tool_calls[tc.tool_name] += 1
                if tc.is_error:
                    tool_errors[tc.tool_name] += 1

        tool_error_rates: list[tuple[str, float, int]] = []
        for tool_name, count in tool_calls.most_common():
            errs = tool_errors.get(tool_name, 0)
            rate = errs / count if count > 0 else 0.0
            tool_error_rates.append((tool_name, rate, count))

        tool_error_rates.sort(key=lambda x: x[1], reverse=True)

        most_failed = tool_error_rates[0] if tool_error_rates else ("none", 0.0, 0)
        avg_latency = total_latency / len(model_sessions) if model_sessions else 0.0

        return ModelReport(
            model=model,
            sessions=len(model_sessions),
            tool_error_rates=tuple(tool_error_rates),
            avg_latency_ms=avg_latency,
            most_failed_tool=most_failed[0],
            most_failed_rate=most_failed[1],
        )

    def track_evolution_impact(
        self,
        before_sessions: list[SessionTrace],
        after_sessions: list[SessionTrace],
        artifact_id: str,
    ) -> ImpactReport:
        """Compare error rates before and after a mutation was applied."""
        before_errors, before_total = self._count_tool_errors(before_sessions, artifact_id)
        after_errors, after_total = self._count_tool_errors(after_sessions, artifact_id)

        before_rate = before_errors / before_total if before_total > 0 else 0.0
        after_rate = after_errors / after_total if after_total > 0 else 0.0
        delta = after_rate - before_rate

        return ImpactReport(
            artifact_id=artifact_id,
            before_error_rate=before_rate,
            after_error_rate=after_rate,
            delta=delta,
            before_sessions=len(before_sessions),
            after_sessions=len(after_sessions),
            improved=delta < 0,
        )

    def detect_regression(
        self,
        before_sessions: list[SessionTrace],
        after_sessions: list[SessionTrace],
        threshold: float = 0.1,
    ) -> list[RegressionAlert]:
        """Detect tools that regressed after mutations were applied."""
        before_rates = self._tool_error_rates(before_sessions)
        after_rates = self._tool_error_rates(after_sessions)

        alerts: list[RegressionAlert] = []
        for tool_name, after_rate in after_rates.items():
            before_rate = before_rates.get(tool_name, 0.0)
            delta = after_rate - before_rate

            if delta > threshold:
                if delta > 0.3:
                    recommendation = "rollback"
                elif delta > 0.15:
                    recommendation = "investigate"
                else:
                    recommendation = "monitor"

                alerts.append(
                    RegressionAlert(
                        artifact_id=tool_name,
                        record_id="",  # Filled by caller if known
                        before_error_rate=before_rate,
                        after_error_rate=after_rate,
                        delta=delta,
                        recommendation=recommendation,
                    )
                )

        alerts.sort(key=lambda a: a.delta, reverse=True)
        return alerts

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _count_tool_errors(sessions: list[SessionTrace], tool_name: str) -> tuple[int, int]:
        """Count errors and total calls for a specific tool."""
        errors = 0
        total = 0
        for session in sessions:
            for tc in session.tool_calls:
                if tc.tool_name == tool_name:
                    total += 1
                    if tc.is_error:
                        errors += 1
        return errors, total

    @staticmethod
    def _tool_error_rates(sessions: list[SessionTrace]) -> dict[str, float]:
        """Compute error rates per tool across sessions."""
        calls: Counter[str] = Counter()
        errors: Counter[str] = Counter()

        for session in sessions:
            for tc in session.tool_calls:
                calls[tc.tool_name] += 1
                if tc.is_error:
                    errors[tc.tool_name] += 1

        return {tool: errors.get(tool, 0) / count for tool, count in calls.items() if count > 0}
