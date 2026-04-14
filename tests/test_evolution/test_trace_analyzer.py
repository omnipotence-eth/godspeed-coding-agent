"""Tests for the trace analyzer — parsing audit JSONL into actionable insights."""

from __future__ import annotations

import gzip
from pathlib import Path

import pytest

from godspeed.audit.events import AuditEventType, AuditRecord
from godspeed.evolution.trace_analyzer import (
    EvolutionReport,
    SessionTrace,
    ToolCall,
    TraceAnalyzer,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_record(
    session_id: str = "sess-1",
    action_type: AuditEventType = AuditEventType.TOOL_CALL,
    detail: dict | None = None,
    outcome: str = "success",
    sequence: int = 0,
) -> str:
    """Create an audit record JSON line."""
    rec = AuditRecord(
        session_id=session_id,
        sequence=sequence,
        action_type=action_type,
        action_detail=detail or {},
        outcome=outcome,
    )
    return rec.model_dump_json()


def _write_session(
    tmp_path: Path,
    session_id: str,
    records: list[str],
    compressed: bool = False,
) -> Path:
    """Write a session audit file from record JSON strings."""
    content = "\n".join(records) + "\n"
    if compressed:
        path = tmp_path / f"{session_id}.audit.jsonl.gz"
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write(content)
    else:
        path = tmp_path / f"{session_id}.audit.jsonl"
        path.write_text(content, encoding="utf-8")
    return path


def _make_tool_call_response_pair(
    session_id: str = "sess-1",
    tool_name: str = "file_read",
    arguments: dict | None = None,
    outcome: str = "success",
    latency_ms: float = 50.0,
    output_length: int = 100,
    seq_start: int = 0,
) -> list[str]:
    """Create a TOOL_CALL + TOOL_RESPONSE pair."""
    call = _make_record(
        session_id=session_id,
        action_type=AuditEventType.TOOL_CALL,
        detail={"tool_name": tool_name, "arguments": arguments or {"path": "src/foo.py"}},
        sequence=seq_start,
    )
    response = _make_record(
        session_id=session_id,
        action_type=AuditEventType.TOOL_RESPONSE,
        detail={
            "tool_name": tool_name,
            "latency_ms": latency_ms,
            "output_length": output_length,
        },
        outcome=outcome,
        sequence=seq_start + 1,
    )
    return [call, response]


# ---------------------------------------------------------------------------
# Test: load_sessions
# ---------------------------------------------------------------------------


class TestLoadSessions:
    def test_empty_dir(self, tmp_path: Path) -> None:
        analyzer = TraceAnalyzer()
        sessions = analyzer.load_sessions(tmp_path)
        assert sessions == []

    def test_nonexistent_dir(self, tmp_path: Path) -> None:
        analyzer = TraceAnalyzer()
        sessions = analyzer.load_sessions(tmp_path / "nope")
        assert sessions == []

    def test_single_session(self, tmp_path: Path) -> None:
        records = _make_tool_call_response_pair()
        _write_session(tmp_path, "sess-1", records)

        analyzer = TraceAnalyzer()
        sessions = analyzer.load_sessions(tmp_path)
        assert len(sessions) == 1
        assert sessions[0].session_id == "sess-1"
        assert len(sessions[0].tool_calls) == 1
        assert sessions[0].tool_calls[0].tool_name == "file_read"

    def test_compressed_session(self, tmp_path: Path) -> None:
        records = _make_tool_call_response_pair(session_id="sess-gz")
        _write_session(tmp_path, "sess-gz", records, compressed=True)

        analyzer = TraceAnalyzer()
        sessions = analyzer.load_sessions(tmp_path)
        assert len(sessions) == 1
        assert sessions[0].session_id == "sess-gz"

    def test_last_n_filter(self, tmp_path: Path) -> None:
        import time

        for i in range(5):
            records = _make_tool_call_response_pair(session_id=f"sess-{i}")
            _write_session(tmp_path, f"sess-{i}", records)
            # Touch with increasing mtime
            time.sleep(0.01)

        analyzer = TraceAnalyzer()
        sessions = analyzer.load_sessions(tmp_path, last_n=2)
        assert len(sessions) == 2

    def test_malformed_records_skipped(self, tmp_path: Path) -> None:
        records = [
            "not valid json",
            *_make_tool_call_response_pair(),
        ]
        _write_session(tmp_path, "sess-bad", records)

        analyzer = TraceAnalyzer()
        sessions = analyzer.load_sessions(tmp_path)
        assert len(sessions) == 1
        assert len(sessions[0].tool_calls) == 1

    def test_empty_file_skipped(self, tmp_path: Path) -> None:
        (tmp_path / "empty.audit.jsonl").write_text("", encoding="utf-8")

        analyzer = TraceAnalyzer()
        sessions = analyzer.load_sessions(tmp_path)
        assert sessions == []


# ---------------------------------------------------------------------------
# Test: analyze_tool_failures
# ---------------------------------------------------------------------------


class TestAnalyzeToolFailures:
    def test_no_errors(self, tmp_path: Path) -> None:
        records = _make_tool_call_response_pair(outcome="success")
        _write_session(tmp_path, "sess-1", records)

        analyzer = TraceAnalyzer()
        sessions = analyzer.load_sessions(tmp_path)
        failures = analyzer.analyze_tool_failures(sessions)
        assert failures == []

    def test_groups_by_tool_and_category(self, tmp_path: Path) -> None:
        records = [
            *_make_tool_call_response_pair(tool_name="bash", outcome="error", seq_start=0),
            *_make_tool_call_response_pair(tool_name="bash", outcome="error", seq_start=2),
            *_make_tool_call_response_pair(tool_name="file_read", outcome="timeout", seq_start=4),
        ]
        _write_session(tmp_path, "sess-1", records)

        analyzer = TraceAnalyzer()
        sessions = analyzer.load_sessions(tmp_path)
        failures = analyzer.analyze_tool_failures(sessions)

        assert len(failures) == 2
        # Sorted by frequency descending
        assert failures[0].tool_name == "bash"
        assert failures[0].frequency == 2
        assert failures[1].tool_name == "file_read"
        assert failures[1].error_category == "timeout"

    def test_example_args_capped_at_5(self, tmp_path: Path) -> None:
        records = []
        for i in range(10):
            records.extend(
                _make_tool_call_response_pair(
                    tool_name="bash",
                    arguments={"cmd": f"cmd-{i}"},
                    outcome="error",
                    seq_start=i * 2,
                )
            )
        _write_session(tmp_path, "sess-1", records)

        analyzer = TraceAnalyzer()
        sessions = analyzer.load_sessions(tmp_path)
        failures = analyzer.analyze_tool_failures(sessions)

        assert failures[0].frequency == 10
        assert len(failures[0].example_args) == 5

    def test_suggested_fix_present(self, tmp_path: Path) -> None:
        records = _make_tool_call_response_pair(tool_name="bash", outcome="timeout")
        _write_session(tmp_path, "sess-1", records)

        analyzer = TraceAnalyzer()
        sessions = analyzer.load_sessions(tmp_path)
        failures = analyzer.analyze_tool_failures(sessions)
        fix = failures[0].suggested_fix.lower()
        assert "timeout" in fix or "background" in fix


# ---------------------------------------------------------------------------
# Test: analyze_tool_latency
# ---------------------------------------------------------------------------


class TestAnalyzeToolLatency:
    def test_basic_latency_stats(self, tmp_path: Path) -> None:
        records = []
        for i, latency in enumerate([10.0, 20.0, 30.0, 40.0, 50.0]):
            records.extend(
                _make_tool_call_response_pair(
                    tool_name="file_read",
                    latency_ms=latency,
                    seq_start=i * 2,
                )
            )
        _write_session(tmp_path, "sess-1", records)

        analyzer = TraceAnalyzer()
        sessions = analyzer.load_sessions(tmp_path)
        stats = analyzer.analyze_tool_latency(sessions)

        assert len(stats) == 1
        assert stats[0].tool_name == "file_read"
        assert stats[0].count == 5
        assert stats[0].mean_ms == 30.0
        assert stats[0].p50_ms == 30.0

    def test_multiple_tools(self, tmp_path: Path) -> None:
        records = [
            *_make_tool_call_response_pair(tool_name="file_read", latency_ms=100.0, seq_start=0),
            *_make_tool_call_response_pair(tool_name="bash", latency_ms=500.0, seq_start=2),
        ]
        _write_session(tmp_path, "sess-1", records)

        analyzer = TraceAnalyzer()
        sessions = analyzer.load_sessions(tmp_path)
        stats = analyzer.analyze_tool_latency(sessions)

        assert len(stats) == 2
        # Sorted by p95 descending
        assert stats[0].tool_name == "bash"

    def test_zero_latency_excluded(self, tmp_path: Path) -> None:
        records = _make_tool_call_response_pair(tool_name="file_read", latency_ms=0.0)
        _write_session(tmp_path, "sess-1", records)

        analyzer = TraceAnalyzer()
        sessions = analyzer.load_sessions(tmp_path)
        stats = analyzer.analyze_tool_latency(sessions)
        assert stats == []


# ---------------------------------------------------------------------------
# Test: analyze_permission_patterns
# ---------------------------------------------------------------------------


class TestAnalyzePermissionPatterns:
    def test_frequent_denials_flagged(self, tmp_path: Path) -> None:
        records = []
        for i in range(6):
            records.append(
                _make_record(
                    action_type=AuditEventType.PERMISSION_CHECK,
                    detail={"tool_name": "bash", "reason": "not in allowlist"},
                    outcome="denied",
                    sequence=i,
                )
            )
        _write_session(tmp_path, "sess-1", records)

        analyzer = TraceAnalyzer()
        sessions = analyzer.load_sessions(tmp_path)
        insights = analyzer.analyze_permission_patterns(sessions)

        assert len(insights) == 1
        assert insights[0].tool_name == "bash"
        assert insights[0].denial_count == 6
        assert insights[0].suggestion == "review_needed"

    def test_frequent_grants_suggest_pre_approve(self, tmp_path: Path) -> None:
        records = []
        for i in range(6):
            records.append(
                _make_record(
                    action_type=AuditEventType.PERMISSION_GRANT,
                    detail={"tool_name": "file_read"},
                    sequence=i,
                )
            )
        _write_session(tmp_path, "sess-1", records)

        analyzer = TraceAnalyzer()
        sessions = analyzer.load_sessions(tmp_path)
        insights = analyzer.analyze_permission_patterns(sessions)

        assert len(insights) == 1
        assert insights[0].suggestion == "pre_approve"

    def test_low_signal_excluded(self, tmp_path: Path) -> None:
        records = [
            _make_record(
                action_type=AuditEventType.PERMISSION_CHECK,
                detail={"tool_name": "bash"},
                outcome="denied",
                sequence=0,
            ),
        ]
        _write_session(tmp_path, "sess-1", records)

        analyzer = TraceAnalyzer()
        sessions = analyzer.load_sessions(tmp_path)
        insights = analyzer.analyze_permission_patterns(sessions)
        assert insights == []


# ---------------------------------------------------------------------------
# Test: analyze_multi_tool_sequences
# ---------------------------------------------------------------------------


class TestAnalyzeMultiToolSequences:
    def test_detects_repeated_pair(self, tmp_path: Path) -> None:
        # Create 3 sessions each with file_read -> file_edit sequence
        for s in range(3):
            records = [
                *_make_tool_call_response_pair(
                    session_id=f"sess-{s}", tool_name="file_read", seq_start=0
                ),
                *_make_tool_call_response_pair(
                    session_id=f"sess-{s}", tool_name="file_edit", seq_start=2
                ),
            ]
            _write_session(tmp_path, f"sess-{s}", records)

        analyzer = TraceAnalyzer()
        sessions = analyzer.load_sessions(tmp_path)
        sequences = analyzer.analyze_multi_tool_sequences(sessions, min_frequency=3)

        tool_pairs = [s.tools for s in sequences]
        assert ("file_read", "file_edit") in tool_pairs

    def test_min_frequency_filter(self, tmp_path: Path) -> None:
        # Only 2 sessions — below threshold of 3
        for s in range(2):
            records = [
                *_make_tool_call_response_pair(
                    session_id=f"sess-{s}", tool_name="file_read", seq_start=0
                ),
                *_make_tool_call_response_pair(
                    session_id=f"sess-{s}", tool_name="file_edit", seq_start=2
                ),
            ]
            _write_session(tmp_path, f"sess-{s}", records)

        analyzer = TraceAnalyzer()
        sessions = analyzer.load_sessions(tmp_path)
        sequences = analyzer.analyze_multi_tool_sequences(sessions, min_frequency=3)
        assert sequences == []

    def test_skill_name_generated(self, tmp_path: Path) -> None:
        for s in range(3):
            records = [
                *_make_tool_call_response_pair(
                    session_id=f"sess-{s}", tool_name="file_read", seq_start=0
                ),
                *_make_tool_call_response_pair(
                    session_id=f"sess-{s}", tool_name="bash", seq_start=2
                ),
            ]
            _write_session(tmp_path, f"sess-{s}", records)

        analyzer = TraceAnalyzer()
        sessions = analyzer.load_sessions(tmp_path)
        sequences = analyzer.analyze_multi_tool_sequences(sessions, min_frequency=3)

        pair = next(s for s in sequences if s.tools == ("file_read", "bash"))
        assert "file_read" in pair.candidate_skill_name
        assert "bash" in pair.candidate_skill_name


# ---------------------------------------------------------------------------
# Test: generate_report
# ---------------------------------------------------------------------------


class TestGenerateReport:
    def test_full_report(self, tmp_path: Path) -> None:
        records = [
            *_make_tool_call_response_pair(tool_name="file_read", seq_start=0),
            *_make_tool_call_response_pair(tool_name="bash", outcome="error", seq_start=2),
        ]
        _write_session(tmp_path, "sess-1", records)

        analyzer = TraceAnalyzer()
        sessions = analyzer.load_sessions(tmp_path)
        report = analyzer.generate_report(sessions)

        assert isinstance(report, EvolutionReport)
        assert report.sessions_analyzed == 1
        assert report.error_rate == 0.5  # 1 error out of 2 calls
        assert len(report.most_used_tools) == 2

    def test_empty_sessions(self) -> None:
        analyzer = TraceAnalyzer()
        report = analyzer.generate_report([])

        assert report.sessions_analyzed == 0
        assert report.error_rate == 0.0
        assert report.tool_failures == ()


# ---------------------------------------------------------------------------
# Test: data structures
# ---------------------------------------------------------------------------


class TestDataStructures:
    def test_tool_call_frozen(self) -> None:
        tc = ToolCall(
            tool_name="bash",
            arguments={"cmd": "ls"},
            output_length=10,
            is_error=False,
            latency_ms=5.0,
            outcome="success",
        )
        with pytest.raises(AttributeError):
            tc.tool_name = "modified"  # type: ignore[misc]

    def test_session_trace_frozen(self) -> None:
        trace = SessionTrace(
            session_id="s1",
            tool_calls=(),
            errors=(),
            permission_denials=(),
            permission_grants=(),
            total_latency_ms=0.0,
            model="",
        )
        with pytest.raises(AttributeError):
            trace.session_id = "modified"  # type: ignore[misc]

    def test_evolution_report_frozen(self) -> None:
        report = EvolutionReport(
            sessions_analyzed=0,
            tool_failures=(),
            latency_stats=(),
            permission_insights=(),
            tool_sequences=(),
            most_used_tools=(),
            error_rate=0.0,
        )
        with pytest.raises(AttributeError):
            report.sessions_analyzed = 5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Test: edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_session_with_only_llm_records(self, tmp_path: Path) -> None:
        records = [
            _make_record(
                action_type=AuditEventType.LLM_REQUEST,
                detail={"model": "claude-sonnet-4-20250514"},
            ),
            _make_record(
                action_type=AuditEventType.LLM_RESPONSE,
                detail={"tokens": 500},
            ),
        ]
        _write_session(tmp_path, "sess-llm", records)

        analyzer = TraceAnalyzer()
        sessions = analyzer.load_sessions(tmp_path)
        assert len(sessions) == 1
        assert sessions[0].model == "claude-sonnet-4-20250514"
        assert sessions[0].tool_calls == ()

    def test_model_extracted_from_llm_request(self, tmp_path: Path) -> None:
        records = [
            _make_record(
                action_type=AuditEventType.LLM_REQUEST,
                detail={"model": "ollama/gemma3:12b"},
            ),
            *_make_tool_call_response_pair(seq_start=1),
        ]
        _write_session(tmp_path, "sess-model", records)

        analyzer = TraceAnalyzer()
        sessions = analyzer.load_sessions(tmp_path)
        assert sessions[0].model == "ollama/gemma3:12b"

    def test_mixed_compressed_and_plain(self, tmp_path: Path) -> None:
        records1 = _make_tool_call_response_pair(session_id="s-plain")
        records2 = _make_tool_call_response_pair(session_id="s-gz")
        _write_session(tmp_path, "s-plain", records1, compressed=False)
        _write_session(tmp_path, "s-gz", records2, compressed=True)

        analyzer = TraceAnalyzer()
        sessions = analyzer.load_sessions(tmp_path)
        assert len(sessions) == 2
        ids = {s.session_id for s in sessions}
        assert ids == {"s-plain", "s-gz"}
