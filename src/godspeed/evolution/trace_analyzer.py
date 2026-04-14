"""Trace analyzer — extract actionable insights from audit trail JSONL.

Parses session audit logs into structured traces and analyzes them for:
- Tool failure patterns (grouped by tool + error category)
- Tool latency statistics (p50/p95/p99)
- Permission patterns (repeatedly denied/approved tools)
- Multi-tool sequences (candidates for skill auto-generation)
"""

from __future__ import annotations

import dataclasses
import gzip
import json
import logging
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from godspeed.audit.events import AuditEventType, AuditRecord

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class ToolCall:
    """A single tool call extracted from audit records."""

    tool_name: str
    arguments: dict[str, Any]
    output_length: int
    is_error: bool
    latency_ms: float
    outcome: str


@dataclasses.dataclass(frozen=True, slots=True)
class SessionTrace:
    """Structured trace for one agent session."""

    session_id: str
    tool_calls: tuple[ToolCall, ...]
    errors: tuple[ToolCall, ...]
    permission_denials: tuple[tuple[str, str], ...]  # (tool_name, reason)
    permission_grants: tuple[str, ...]  # tool_name
    total_latency_ms: float
    model: str


@dataclasses.dataclass(frozen=True, slots=True)
class ToolFailurePattern:
    """A recurring tool failure pattern across sessions."""

    tool_name: str
    error_category: str  # "invalid_args", "permission_denied", "timeout", "execution_error"
    frequency: int
    example_args: tuple[dict[str, Any], ...]
    suggested_fix: str


@dataclasses.dataclass(frozen=True, slots=True)
class LatencyStats:
    """Latency statistics for a single tool."""

    tool_name: str
    count: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float


@dataclasses.dataclass(frozen=True, slots=True)
class PermissionInsight:
    """Insight about a tool's permission patterns."""

    tool_name: str
    denial_count: int
    grant_count: int
    suggestion: str  # "add_to_allowlist" | "pre_approve" | "review_needed"


@dataclasses.dataclass(frozen=True, slots=True)
class ToolSequence:
    """A repeated multi-tool pattern detected across sessions."""

    tools: tuple[str, ...]
    frequency: int
    avg_success_rate: float
    candidate_skill_name: str


@dataclasses.dataclass(frozen=True, slots=True)
class EvolutionReport:
    """Aggregated analysis report for the mutation engine."""

    sessions_analyzed: int
    tool_failures: tuple[ToolFailurePattern, ...]
    latency_stats: tuple[LatencyStats, ...]
    permission_insights: tuple[PermissionInsight, ...]
    tool_sequences: tuple[ToolSequence, ...]
    most_used_tools: tuple[tuple[str, int], ...]
    error_rate: float  # overall error rate across all tool calls


# ---------------------------------------------------------------------------
# Trace Analyzer
# ---------------------------------------------------------------------------


class TraceAnalyzer:
    """Parse audit trail JSONL into actionable insights."""

    def load_sessions(self, audit_dir: Path, last_n: int = 0) -> list[SessionTrace]:
        """Load session traces from audit JSONL files.

        Args:
            audit_dir: Directory containing .audit.jsonl and .audit.jsonl.gz files.
            last_n: If > 0, only return the most recent N sessions (by file mtime).

        Returns:
            List of SessionTrace objects, one per session file.
        """
        if not audit_dir.is_dir():
            logger.warning("Audit directory does not exist path=%s", audit_dir)
            return []

        # Collect all audit files
        files: list[Path] = []
        for pattern in ("*.audit.jsonl", "*.audit.jsonl.gz"):
            files.extend(audit_dir.glob(pattern))

        if not files:
            return []

        # Sort by modification time (newest first) for last_n filtering
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        if last_n > 0:
            files = files[:last_n]

        sessions: list[SessionTrace] = []
        for path in files:
            trace = self._parse_session(path)
            if trace is not None:
                sessions.append(trace)

        return sessions

    def analyze_tool_failures(self, sessions: list[SessionTrace]) -> list[ToolFailurePattern]:
        """Group errors by tool_name + error category, rank by frequency."""
        # Key: (tool_name, error_category) -> list of example args
        failures: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

        for session in sessions:
            for tc in session.errors:
                category = self._classify_error(tc)
                failures[(tc.tool_name, category)].append(tc.arguments)

        patterns: list[ToolFailurePattern] = []
        for (tool_name, category), args_list in failures.items():
            examples = tuple(args_list[:5])  # Keep at most 5 examples
            suggested_fix = self._suggest_fix(tool_name, category, examples)
            patterns.append(
                ToolFailurePattern(
                    tool_name=tool_name,
                    error_category=category,
                    frequency=len(args_list),
                    example_args=examples,
                    suggested_fix=suggested_fix,
                )
            )

        # Sort by frequency descending
        patterns.sort(key=lambda p: p.frequency, reverse=True)
        return patterns

    def analyze_tool_latency(self, sessions: list[SessionTrace]) -> list[LatencyStats]:
        """Compute p50/p95/p99 latency per tool across all sessions."""
        latencies: dict[str, list[float]] = defaultdict(list)

        for session in sessions:
            for tc in session.tool_calls:
                if tc.latency_ms > 0:
                    latencies[tc.tool_name].append(tc.latency_ms)

        stats: list[LatencyStats] = []
        for tool_name, values in latencies.items():
            if not values:
                continue
            sorted_vals = sorted(values)
            n = len(sorted_vals)
            stats.append(
                LatencyStats(
                    tool_name=tool_name,
                    count=n,
                    p50_ms=self._percentile(sorted_vals, 50),
                    p95_ms=self._percentile(sorted_vals, 95),
                    p99_ms=self._percentile(sorted_vals, 99),
                    mean_ms=statistics.mean(sorted_vals),
                )
            )

        stats.sort(key=lambda s: s.p95_ms, reverse=True)
        return stats

    def analyze_permission_patterns(self, sessions: list[SessionTrace]) -> list[PermissionInsight]:
        """Identify tools that are repeatedly denied or always approved."""
        denials: Counter[str] = Counter()
        grants: Counter[str] = Counter()

        for session in sessions:
            for tool_name, _reason in session.permission_denials:
                denials[tool_name] += 1
            for tool_name in session.permission_grants:
                grants[tool_name] += 1

        all_tools = set(denials.keys()) | set(grants.keys())
        insights: list[PermissionInsight] = []

        for tool_name in all_tools:
            d = denials.get(tool_name, 0)
            g = grants.get(tool_name, 0)

            if d >= 5 and g == 0:
                suggestion = "review_needed"
            elif d >= 5:
                suggestion = "add_to_allowlist"
            elif g >= 5 and d == 0:
                suggestion = "pre_approve"
            else:
                continue  # Not enough signal

            insights.append(
                PermissionInsight(
                    tool_name=tool_name,
                    denial_count=d,
                    grant_count=g,
                    suggestion=suggestion,
                )
            )

        insights.sort(key=lambda i: i.denial_count + i.grant_count, reverse=True)
        return insights

    def analyze_multi_tool_sequences(
        self, sessions: list[SessionTrace], min_frequency: int = 3
    ) -> list[ToolSequence]:
        """Detect repeated tool chains across sessions."""
        # Extract tool name sequences per session (sliding windows of 2-4)
        sequence_counts: Counter[tuple[str, ...]] = Counter()
        sequence_successes: dict[tuple[str, ...], list[bool]] = defaultdict(list)

        for session in sessions:
            names = [tc.tool_name for tc in session.tool_calls]
            errors = {i for i, tc in enumerate(session.tool_calls) if tc.is_error}

            for window_size in (2, 3, 4):
                for i in range(len(names) - window_size + 1):
                    seq = tuple(names[i : i + window_size])
                    sequence_counts[seq] += 1
                    all_ok = all(j not in errors for j in range(i, i + window_size))
                    sequence_successes[seq].append(all_ok)

        results: list[ToolSequence] = []
        for seq, count in sequence_counts.items():
            if count < min_frequency:
                continue
            successes = sequence_successes[seq]
            avg_success = sum(successes) / len(successes) if successes else 0.0
            skill_name = "_and_".join(dict.fromkeys(seq))  # Deduplicate consecutive
            results.append(
                ToolSequence(
                    tools=seq,
                    frequency=count,
                    avg_success_rate=avg_success,
                    candidate_skill_name=skill_name,
                )
            )

        results.sort(key=lambda r: r.frequency, reverse=True)
        return results

    def generate_report(self, sessions: list[SessionTrace]) -> EvolutionReport:
        """Generate a full analysis report from session traces."""
        # Tool usage counts
        tool_counts: Counter[str] = Counter()
        total_calls = 0
        total_errors = 0

        for session in sessions:
            for tc in session.tool_calls:
                tool_counts[tc.tool_name] += 1
                total_calls += 1
                if tc.is_error:
                    total_errors += 1

        error_rate = total_errors / total_calls if total_calls > 0 else 0.0

        return EvolutionReport(
            sessions_analyzed=len(sessions),
            tool_failures=tuple(self.analyze_tool_failures(sessions)),
            latency_stats=tuple(self.analyze_tool_latency(sessions)),
            permission_insights=tuple(self.analyze_permission_patterns(sessions)),
            tool_sequences=tuple(self.analyze_multi_tool_sequences(sessions)),
            most_used_tools=tuple(tool_counts.most_common(10)),
            error_rate=error_rate,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_session(self, path: Path) -> SessionTrace | None:
        """Parse a single audit JSONL file into a SessionTrace.

        Reads line-by-line (streaming) to avoid loading entire files into memory.
        This matters on constrained devices (Jetson Orin Nano, 8GB shared RAM).
        """
        records: list[AuditRecord] = []
        try:
            if str(path).endswith(".gz"):
                ctx = gzip.open(path, "rt", encoding="utf-8")  # noqa: SIM115
            else:
                ctx = open(path, encoding="utf-8")  # noqa: SIM115
            with ctx as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        records.append(AuditRecord.model_validate(data))
                    except (json.JSONDecodeError, Exception) as exc:
                        logger.debug(
                            "Skipping malformed audit record path=%s error=%s",
                            path,
                            exc,
                        )
                        continue
        except (OSError, gzip.BadGzipFile) as exc:
            logger.warning("Failed to read audit file path=%s error=%s", path, exc)
            return None

        if not records:
            return None

        session_id = records[0].session_id
        model = ""

        tool_calls: list[ToolCall] = []
        permission_denials: list[tuple[str, str]] = []
        permission_grants: list[str] = []
        total_latency = 0.0

        # Build tool call / response pairs
        pending_calls: dict[str, AuditRecord] = {}  # tool_name+seq -> call record

        for rec in records:
            if rec.action_type == AuditEventType.TOOL_CALL:
                tool_name = rec.action_detail.get("tool_name", "unknown")
                pending_calls[f"{tool_name}_{rec.sequence}"] = rec

            elif rec.action_type == AuditEventType.TOOL_RESPONSE:
                tool_name = rec.action_detail.get("tool_name", "unknown")
                is_error = rec.outcome in ("error", "timeout")
                latency_ms = rec.action_detail.get("latency_ms", 0.0)
                output_length = rec.action_detail.get("output_length", 0)
                arguments = rec.action_detail.get("arguments", {})

                # Try to find the matching call for arguments
                for key, call_rec in list(pending_calls.items()):
                    if key.startswith(f"{tool_name}_"):
                        arguments = call_rec.action_detail.get("arguments", arguments)
                        del pending_calls[key]
                        break

                tc = ToolCall(
                    tool_name=tool_name,
                    arguments=arguments,
                    output_length=output_length,
                    is_error=is_error,
                    latency_ms=latency_ms,
                    outcome=rec.outcome,
                )
                tool_calls.append(tc)
                total_latency += latency_ms

            elif rec.action_type == AuditEventType.PERMISSION_CHECK:
                tool_name = rec.action_detail.get("tool_name", "unknown")
                if rec.outcome == "denied":
                    reason = rec.action_detail.get("reason", "unknown")
                    permission_denials.append((tool_name, reason))

            elif rec.action_type == AuditEventType.PERMISSION_GRANT:
                tool_name = rec.action_detail.get("tool_name", "unknown")
                permission_grants.append(tool_name)

            elif rec.action_type == AuditEventType.LLM_REQUEST:
                model = rec.action_detail.get("model", model)

        errors = tuple(tc for tc in tool_calls if tc.is_error)

        return SessionTrace(
            session_id=session_id,
            tool_calls=tuple(tool_calls),
            errors=errors,
            permission_denials=tuple(permission_denials),
            permission_grants=tuple(permission_grants),
            total_latency_ms=total_latency,
            model=model,
        )

    @staticmethod
    def _classify_error(tc: ToolCall) -> str:
        """Classify a tool call error into a category."""
        if tc.outcome == "timeout":
            return "timeout"
        if tc.outcome == "denied":
            return "permission_denied"
        # Check arguments for common patterns
        args_str = str(tc.arguments).lower()
        if "not found" in args_str or "no such file" in args_str:
            return "invalid_args"
        return "execution_error"

    @staticmethod
    def _suggest_fix(tool_name: str, category: str, examples: tuple[dict[str, Any], ...]) -> str:
        """Generate a human-readable fix suggestion for a failure pattern."""
        suggestions = {
            "invalid_args": (
                f"Tool '{tool_name}' description may be unclear about valid argument values"
            ),
            "permission_denied": (f"Tool '{tool_name}' needs permission configuration review"),
            "timeout": (f"Tool '{tool_name}' may need a higher timeout or background execution"),
            "execution_error": (f"Tool '{tool_name}' description may need better examples"),
        }
        return suggestions.get(category, f"Review tool '{tool_name}' for recurring errors")

    @staticmethod
    def _percentile(sorted_values: list[float], pct: int) -> float:
        """Compute the p-th percentile of a sorted list."""
        if not sorted_values:
            return 0.0
        n = len(sorted_values)
        idx = (pct / 100) * (n - 1)
        lower = int(idx)
        upper = min(lower + 1, n - 1)
        frac = idx - lower
        return sorted_values[lower] * (1 - frac) + sorted_values[upper] * frac
