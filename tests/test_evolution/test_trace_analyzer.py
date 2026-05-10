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
            "also not json\n",
            "",
            *_make_tool_call_response_pair(),
        ]
        _write_session(tmp_path, "sess-bad", records)

        analyzer = TraceAnalyzer()
        sessions = analyzer.load_sessions(tmp_path)
        assert len(sessions) == 1
        assert len(sessions[0].tool_calls) == 1

    def test_empty_lines_skipped(self, tmp_path: Path) -> None:
        records = [
            "",
            "\n",
            "   \n",
            *_make_tool_call_response_pair(),
            "",
        ]
        _write_session(tmp_path, "sess-blank", records)

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
        assert stats[0].tool_name == "bash"

    def test_zero_latency_excluded(self, tmp_path: Path) -> None:
        records = _make_tool_call_response_pair(tool_name="file_read", latency_ms=0.0)
        _write_session(tmp_path, "sess-1", records)

        analyzer = TraceAnalyzer()
        sessions = analyzer.load_sessions(tmp_path)
        stats = analyzer.analyze_tool_latency(sessions)
        assert stats == []

    def test_mixed_latencies_some_zero(self, tmp_path: Path) -> None:
        records = [
            *_make_tool_call_response_pair(tool_name="file_read", latency_ms=100.0, seq_start=0),
            *_make_tool_call_response_pair(tool_name="file_read", latency_ms=0.0, seq_start=2),
            *_make_tool_call_response_pair(tool_name="bash", latency_ms=0.0, seq_start=4),
        ]
        _write_session(tmp_path, "sess-1", records)

        analyzer = TraceAnalyzer()
        sessions = analyzer.load_sessions(tmp_path)
        stats = analyzer.analyze_tool_latency(sessions)

        # file_read has one non-zero latency, bash has zero
        file_read_stats = [s for s in stats if s.tool_name == "file_read"]
        bash_stats = [s for s in stats if s.tool_name == "bash"]
        assert len(file_read_stats) == 1
        assert len(bash_stats) == 0


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


# ---------------------------------------------------------------------------
# Test: _parse_session edge cases
# ---------------------------------------------------------------------------


class TestParseSessionEdgeCases:
    def test_corrupt_gzip_file(self, tmp_path: Path) -> None:
        path = tmp_path / "corrupt.audit.jsonl.gz"
        path.write_bytes(b"not a valid gzip file")

        analyzer = TraceAnalyzer()
        sessions = analyzer.load_sessions(tmp_path)
        assert sessions == []

    def test_permission_denial_classification(self, tmp_path: Path) -> None:
        records = _make_tool_call_response_pair(
            tool_name="bash",
            arguments={"cmd": "rm -rf /"},
            outcome="error",
        )
        _write_session(tmp_path, "sess-1", records)

        analyzer = TraceAnalyzer()
        sessions = analyzer.load_sessions(tmp_path)
        failures = analyzer.analyze_tool_failures(sessions)
        assert len(failures) == 1

    def test_timeout_classification(self, tmp_path: Path) -> None:
        records = _make_tool_call_response_pair(
            tool_name="bash",
            outcome="timeout",
        )
        _write_session(tmp_path, "sess-1", records)

        analyzer = TraceAnalyzer()
        sessions = analyzer.load_sessions(tmp_path)
        failures = analyzer.analyze_tool_failures(sessions)
        assert len(failures) == 1
        assert failures[0].error_category == "timeout"

    def test_invalid_args_classification(self, tmp_path: Path) -> None:
        records = _make_tool_call_response_pair(
            tool_name="file_read",
            arguments={"path": "file not found"},
            outcome="error",
        )
        _write_session(tmp_path, "sess-1", records)

        analyzer = TraceAnalyzer()
        sessions = analyzer.load_sessions(tmp_path)
        failures = analyzer.analyze_tool_failures(sessions)
        assert len(failures) == 1
        assert failures[0].error_category == "invalid_args"

    def test_execution_error_classification(self, tmp_path: Path) -> None:
        records = _make_tool_call_response_pair(
            tool_name="file_edit",
            arguments={"path": "foo.py", "content": "x"},
            outcome="error",
        )
        _write_session(tmp_path, "sess-1", records)

        analyzer = TraceAnalyzer()
        sessions = analyzer.load_sessions(tmp_path)
        failures = analyzer.analyze_tool_failures(sessions)
        assert len(failures) == 1
        assert failures[0].error_category == "execution_error"

    def test_tool_response_without_call(self, tmp_path: Path) -> None:
        records = [
            _make_record(
                action_type=AuditEventType.TOOL_RESPONSE,
                detail={
                    "tool_name": "bash",
                    "latency_ms": 10.0,
                    "output_length": 50,
                },
                outcome="success",
                sequence=0,
            )
        ]
        _write_session(tmp_path, "sess-1", records)

        analyzer = TraceAnalyzer()
        sessions = analyzer.load_sessions(tmp_path)
        assert len(sessions) == 1
        # Arguments should be from the response, not a call
        assert len(sessions[0].tool_calls) == 1


# ---------------------------------------------------------------------------
# Test: Permission insights edge cases
# ---------------------------------------------------------------------------


class TestPermissionInsightsEdgeCases:
    def test_add_to_allowlist_suggestion(self, tmp_path: Path) -> None:
        records = []
        for i in range(5):
            records.append(
                _make_record(
                    action_type=AuditEventType.PERMISSION_CHECK,
                    detail={"tool_name": "bash", "reason": "not in allowlist"},
                    outcome="denied",
                    sequence=i * 2,
                )
            )
            records.append(
                _make_record(
                    action_type=AuditEventType.PERMISSION_GRANT,
                    detail={"tool_name": "bash"},
                    sequence=i * 2 + 1,
                )
            )
        _write_session(tmp_path, "sess-1", records)

        analyzer = TraceAnalyzer()
        sessions = analyzer.load_sessions(tmp_path)
        insights = analyzer.analyze_permission_patterns(sessions)
        assert len(insights) == 1
        assert insights[0].suggestion == "add_to_allowlist"

    def test_mixed_tools_permission(self, tmp_path: Path) -> None:
        records = []
        for _ in range(6):
            records.append(
                _make_record(
                    action_type=AuditEventType.PERMISSION_CHECK,
                    detail={"tool_name": "bash"},
                    outcome="denied",
                    sequence=len(records),
                )
            )
        for _ in range(6):
            records.append(
                _make_record(
                    action_type=AuditEventType.PERMISSION_GRANT,
                    detail={"tool_name": "file_read"},
                    sequence=len(records),
                )
            )
        _write_session(tmp_path, "sess-1", records)

        analyzer = TraceAnalyzer()
        sessions = analyzer.load_sessions(tmp_path)
        insights = analyzer.analyze_permission_patterns(sessions)

        bash_insight = next(i for i in insights if i.tool_name == "bash")
        file_insight = next(i for i in insights if i.tool_name == "file_read")
        assert bash_insight.suggestion == "review_needed"
        assert file_insight.suggestion == "pre_approve"

    def test_exactly_five_grants(self, tmp_path: Path) -> None:
        records = []
        for i in range(5):
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


# ---------------------------------------------------------------------------
# Test: generate_report edge cases
# ---------------------------------------------------------------------------


class TestGenerateReportEdgeCases:
    def test_report_with_empty_latencies(self, tmp_path: Path) -> None:
        records = _make_tool_call_response_pair(tool_name="bash", latency_ms=0.0)
        _write_session(tmp_path, "sess-1", records)

        analyzer = TraceAnalyzer()
        sessions = analyzer.load_sessions(tmp_path)
        report = analyzer.generate_report(sessions)

        assert report.sessions_analyzed == 1
        assert len(report.latency_stats) == 0

    def test_report_with_permissions(self, tmp_path: Path) -> None:
        records = [
            _make_record(
                action_type=AuditEventType.PERMISSION_CHECK,
                detail={"tool_name": "bash", "reason": "not allowed"},
                outcome="denied",
                sequence=0,
            ),
            _make_record(
                action_type=AuditEventType.PERMISSION_CHECK,
                detail={"tool_name": "bash", "reason": "not allowed"},
                outcome="denied",
                sequence=1,
            ),
            _make_record(
                action_type=AuditEventType.PERMISSION_CHECK,
                detail={"tool_name": "bash", "reason": "not allowed"},
                outcome="denied",
                sequence=2,
            ),
            _make_record(
                action_type=AuditEventType.PERMISSION_CHECK,
                detail={"tool_name": "bash", "reason": "not allowed"},
                outcome="denied",
                sequence=3,
            ),
            _make_record(
                action_type=AuditEventType.PERMISSION_CHECK,
                detail={"tool_name": "bash", "reason": "not allowed"},
                outcome="denied",
                sequence=4,
            ),
            *_make_tool_call_response_pair(seq_start=5),
        ]
        _write_session(tmp_path, "sess-1", records)

        analyzer = TraceAnalyzer()
        sessions = analyzer.load_sessions(tmp_path)
        report = analyzer.generate_report(sessions)

        assert report.sessions_analyzed == 1
        assert len(report.permission_insights) >= 1

    def test_report_error_rate_zero_for_no_calls(self, tmp_path: Path) -> None:
        records = [
            _make_record(
                action_type=AuditEventType.LLM_REQUEST,
                detail={"model": "test"},
                sequence=0,
            ),
        ]
        _write_session(tmp_path, "sess-1", records)

        analyzer = TraceAnalyzer()
        sessions = analyzer.load_sessions(tmp_path)
        report = analyzer.generate_report(sessions)

        assert report.error_rate == 0.0
        assert report.tool_failures == ()

    def test_report_with_multi_tool_sequences(self, tmp_path: Path) -> None:
        for s in range(4):
            records = [
                *_make_tool_call_response_pair(
                    session_id=f"sess-{s}", tool_name="file_read", seq_start=0
                ),
                *_make_tool_call_response_pair(
                    session_id=f"sess-{s}", tool_name="file_edit", seq_start=2
                ),
                *_make_tool_call_response_pair(
                    session_id=f"sess-{s}", tool_name="bash", seq_start=4
                ),
            ]
            _write_session(tmp_path, f"sess-{s}", records)

        analyzer = TraceAnalyzer()
        sessions = analyzer.load_sessions(tmp_path)
        report = analyzer.generate_report(sessions)

        assert report.sessions_analyzed == 4
        assert len(report.tool_sequences) >= 1

    def test_report_with_permission_grants_and_denials(self, tmp_path: Path) -> None:
        records = []
        for i in range(5):
            records.append(
                _make_record(
                    action_type=AuditEventType.PERMISSION_GRANT,
                    detail={"tool_name": "bash"},
                    sequence=i,
                )
            )
        for i in range(5, 10):
            records.append(
                _make_record(
                    action_type=AuditEventType.PERMISSION_CHECK,
                    detail={"tool_name": "bash", "reason": "not allowed"},
                    outcome="denied",
                    sequence=i,
                )
            )
        _write_session(tmp_path, "sess-1", records)

        analyzer = TraceAnalyzer()
        sessions = analyzer.load_sessions(tmp_path)
        report = analyzer.generate_report(sessions)

        assert report.sessions_analyzed == 1
        assert len(report.permission_insights) >= 1


# ---------------------------------------------------------------------------
# Test: _percentile
# ---------------------------------------------------------------------------


class TestPercentile:
    def test_basic_percentile(self) -> None:
        vals = [10.0, 20.0, 30.0, 40.0, 50.0]
        p50 = TraceAnalyzer._percentile(vals, 50)
        p95 = TraceAnalyzer._percentile(vals, 95)
        p99 = TraceAnalyzer._percentile(vals, 99)
        assert p50 == 30.0
        assert p95 >= 40.0
        assert p99 >= 40.0

    def test_percentile_single_value(self) -> None:
        assert TraceAnalyzer._percentile([42.0], 50) == 42.0
        assert TraceAnalyzer._percentile([42.0], 95) == 42.0
        assert TraceAnalyzer._percentile([42.0], 99) == 42.0

    def test_percentile_two_values(self) -> None:
        vals = [10.0, 20.0]
        p50 = TraceAnalyzer._percentile(vals, 50)
        p95 = TraceAnalyzer._percentile(vals, 95)
        assert 10.0 <= p50 <= 20.0
        assert 10.0 <= p95 <= 20.0

    def test_percentile_empty_list(self) -> None:
        assert TraceAnalyzer._percentile([], 50) == 0.0

    def test_percentile_at_boundaries(self) -> None:
        vals = [0.0, 100.0]
        p0 = TraceAnalyzer._percentile(vals, 0)
        p100 = TraceAnalyzer._percentile(vals, 100)
        assert p0 == 0.0
        assert p100 == 100.0


# ---------------------------------------------------------------------------
# Test: _classify_error
# ---------------------------------------------------------------------------


class TestClassifyError:
    def test_classify_timeout(self) -> None:
        tc = ToolCall(
            tool_name="bash",
            arguments={"cmd": "ls"},
            output_length=0,
            is_error=True,
            latency_ms=30000.0,
            outcome="timeout",
        )
        assert TraceAnalyzer._classify_error(tc) == "timeout"

    def test_classify_denied(self) -> None:
        tc = ToolCall(
            tool_name="bash",
            arguments={"cmd": "rm"},
            output_length=0,
            is_error=True,
            latency_ms=5.0,
            outcome="denied",
        )
        assert TraceAnalyzer._classify_error(tc) == "permission_denied"

    def test_classify_not_found(self) -> None:
        tc = ToolCall(
            tool_name="file_read",
            arguments={"path": "no such file"},
            output_length=0,
            is_error=True,
            latency_ms=5.0,
            outcome="error",
        )
        assert TraceAnalyzer._classify_error(tc) == "invalid_args"

    def test_classify_execution_error(self) -> None:
        tc = ToolCall(
            tool_name="file_edit",
            arguments={"path": "foo.py", "content": "x"},
            output_length=0,
            is_error=True,
            latency_ms=5.0,
            outcome="error",
        )
        assert TraceAnalyzer._classify_error(tc) == "execution_error"


# ---------------------------------------------------------------------------
# Test: _suggest_fix
# ---------------------------------------------------------------------------


class TestSuggestFix:
    def test_invalid_args_suggestion(self) -> None:
        fix = TraceAnalyzer._suggest_fix("bash", "invalid_args", ())
        assert "description" in fix.lower()

    def test_permission_denied_suggestion(self) -> None:
        fix = TraceAnalyzer._suggest_fix("bash", "permission_denied", ())
        assert "permission" in fix.lower()

    def test_timeout_suggestion(self) -> None:
        fix = TraceAnalyzer._suggest_fix("bash", "timeout", ())
        assert "timeout" in fix.lower()

    def test_execution_error_suggestion(self) -> None:
        fix = TraceAnalyzer._suggest_fix("bash", "execution_error", ())
        assert "example" in fix.lower()

    def test_unknown_category_suggestion(self) -> None:
        fix = TraceAnalyzer._suggest_fix("mytool", "unknown_cat", ())
        assert "mytool" in fix

    def test_no_examples(self) -> None:
        fix = TraceAnalyzer._suggest_fix("mytool", "timeout", ())
        assert "mytool" in fix


# ---------------------------------------------------------------------------
# Test: LatencyStats frozen
# ---------------------------------------------------------------------------


class TestLatencyStats:
    def test_frozen(self) -> None:
        from godspeed.evolution.trace_analyzer import LatencyStats

        ls = LatencyStats(
            tool_name="bash",
            count=5,
            p50_ms=50.0,
            p95_ms=95.0,
            p99_ms=99.0,
            mean_ms=60.0,
        )
        with pytest.raises(AttributeError):
            ls.count = 10  # type: ignore[misc]

    def test_fields(self) -> None:
        from godspeed.evolution.trace_analyzer import LatencyStats

        ls = LatencyStats(
            tool_name="bash",
            count=5,
            p50_ms=50.0,
            p95_ms=95.0,
            p99_ms=99.0,
            mean_ms=60.0,
        )
        assert ls.tool_name == "bash"
        assert ls.mean_ms == 60.0


# ---------------------------------------------------------------------------
# Test: PermissionInsight frozen
# ---------------------------------------------------------------------------


class TestPermissionInsight:
    def test_frozen(self) -> None:
        from godspeed.evolution.trace_analyzer import PermissionInsight

        pi = PermissionInsight(
            tool_name="bash",
            denial_count=5,
            grant_count=0,
            suggestion="review_needed",
        )
        with pytest.raises(AttributeError):
            pi.suggestion = "changed"  # type: ignore[misc]

    def test_fields(self) -> None:
        from godspeed.evolution.trace_analyzer import PermissionInsight

        pi = PermissionInsight(
            tool_name="bash",
            denial_count=3,
            grant_count=7,
            suggestion="add_to_allowlist",
        )
        assert pi.tool_name == "bash"
        assert pi.denial_count == 3
        assert pi.grant_count == 7


# ---------------------------------------------------------------------------
# Test: ToolSequence frozen
# ---------------------------------------------------------------------------


class TestToolSequence:
    def test_frozen(self) -> None:
        from godspeed.evolution.trace_analyzer import ToolSequence

        ts = ToolSequence(
            tools=("file_read", "file_edit"),
            frequency=5,
            avg_success_rate=0.9,
            candidate_skill_name="read_edit",
        )
        with pytest.raises(AttributeError):
            ts.frequency = 10  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Test: ToolFailurePattern frozen
# ---------------------------------------------------------------------------


class TestToolFailurePattern:
    def test_frozen(self) -> None:
        from godspeed.evolution.trace_analyzer import ToolFailurePattern

        tfp = ToolFailurePattern(
            tool_name="bash",
            error_category="timeout",
            frequency=3,
            example_args=(),
            suggested_fix="Increase timeout",
        )
        with pytest.raises(AttributeError):
            tfp.frequency = 5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Test: session comparison and empty trace handling
# ---------------------------------------------------------------------------


class TestSessionComparison:
    def test_multiple_sessions_same_tool_different_outcomes(self, tmp_path: Path) -> None:
        records1 = [
            *_make_tool_call_response_pair(
                session_id="sess-1", tool_name="bash", outcome="success", seq_start=0
            ),
        ]
        records2 = [
            *_make_tool_call_response_pair(
                session_id="sess-2", tool_name="bash", outcome="error", seq_start=0
            ),
        ]
        _write_session(tmp_path, "sess-1", records1)
        _write_session(tmp_path, "sess-2", records2)

        analyzer = TraceAnalyzer()
        sessions = analyzer.load_sessions(tmp_path)
        assert len(sessions) == 2

        failures = analyzer.analyze_tool_failures(sessions)
        assert len(failures) == 1
        assert failures[0].tool_name == "bash"
        assert failures[0].frequency == 1

        report = analyzer.generate_report(sessions)
        assert report.sessions_analyzed == 2
        assert report.error_rate == 0.5


class TestEmptyTraceHandling:
    def test_all_sessions_have_no_tool_calls(self, tmp_path: Path) -> None:
        records = [
            _make_record(
                action_type=AuditEventType.LLM_REQUEST,
                detail={"model": "test"},
                sequence=0,
            ),
            _make_record(
                action_type=AuditEventType.LLM_RESPONSE,
                detail={"tokens": 100},
                sequence=1,
            ),
        ]
        _write_session(tmp_path, "sess-no-tools", records)

        analyzer = TraceAnalyzer()
        sessions = analyzer.load_sessions(tmp_path)
        assert len(sessions) == 1
        assert sessions[0].tool_calls == ()

        report = analyzer.generate_report(sessions)
        assert report.error_rate == 0.0
        assert report.most_used_tools == ()


class TestToolFrequencyWithThresholds:
    def test_varying_min_frequency(self, tmp_path: Path) -> None:
        for s in range(4):
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

        sequences_high = analyzer.analyze_multi_tool_sequences(sessions, min_frequency=5)
        assert len(sequences_high) == 0

        sequences_med = analyzer.analyze_multi_tool_sequences(sessions, min_frequency=3)
        assert len(sequences_med) >= 1

    def test_window_size_three_sequence(self, tmp_path: Path) -> None:
        for s in range(4):
            records = [
                *_make_tool_call_response_pair(
                    session_id=f"sess-{s}", tool_name="file_read", seq_start=0
                ),
                *_make_tool_call_response_pair(
                    session_id=f"sess-{s}", tool_name="file_edit", seq_start=2
                ),
                *_make_tool_call_response_pair(
                    session_id=f"sess-{s}", tool_name="bash", seq_start=4
                ),
            ]
            _write_session(tmp_path, f"sess-{s}", records)

        analyzer = TraceAnalyzer()
        sessions = analyzer.load_sessions(tmp_path)
        sequences = analyzer.analyze_multi_tool_sequences(sessions, min_frequency=3)

        triple = next((s for s in sequences if s.tools == ("file_read", "file_edit", "bash")), None)
        assert triple is not None
        assert triple.frequency >= 3


class TestPerformanceMetricAnalysis:
    def test_latency_stats_include_p99(self, tmp_path: Path) -> None:
        records = []
        for i in range(100):
            records.extend(
                _make_tool_call_response_pair(
                    tool_name="bash",
                    latency_ms=float((i + 1) * 10),
                    seq_start=i * 2,
                )
            )
        _write_session(tmp_path, "sess-1", records)

        analyzer = TraceAnalyzer()
        sessions = analyzer.load_sessions(tmp_path)
        stats = analyzer.analyze_tool_latency(sessions)

        assert len(stats) == 1
        assert stats[0].count == 100
        assert stats[0].p99_ms >= stats[0].p95_ms
        assert stats[0].p95_ms >= stats[0].p50_ms


class TestToolSuccessRateTracking:
    def test_success_rate_from_sequence_errors(self, tmp_path: Path) -> None:
        for s in range(4):
            records = [
                *_make_tool_call_response_pair(
                    session_id=f"sess-{s}", tool_name="file_read", outcome="success", seq_start=0
                ),
                *_make_tool_call_response_pair(
                    session_id=f"sess-{s}", tool_name="bash", outcome="error", seq_start=2
                ),
            ]
            _write_session(tmp_path, f"sess-{s}", records)

        analyzer = TraceAnalyzer()
        sessions = analyzer.load_sessions(tmp_path)
        sequences = analyzer.analyze_multi_tool_sequences(sessions, min_frequency=3)

        pair = next(s for s in sequences if s.tools == ("file_read", "bash"))
        assert pair.avg_success_rate == 0.0

    def test_all_ok_sequence_success_rate(self, tmp_path: Path) -> None:
        for s in range(4):
            records = [
                *_make_tool_call_response_pair(
                    session_id=f"sess-{s}", tool_name="file_read", outcome="success", seq_start=0
                ),
                *_make_tool_call_response_pair(
                    session_id=f"sess-{s}", tool_name="file_edit", outcome="success", seq_start=2
                ),
            ]
            _write_session(tmp_path, f"sess-{s}", records)

        analyzer = TraceAnalyzer()
        sessions = analyzer.load_sessions(tmp_path)
        sequences = analyzer.analyze_multi_tool_sequences(sessions, min_frequency=3)

        pair = next(s for s in sequences if s.tools == ("file_read", "file_edit"))
        assert pair.avg_success_rate == 1.0


class TestParseSessionMoreEdgeCases:
    def test_tool_response_with_matching_call_picks_arguments(self, tmp_path: Path) -> None:
        records = [
            _make_record(
                action_type=AuditEventType.TOOL_CALL,
                detail={"tool_name": "bash", "arguments": {"cmd": "ls -la"}},
                sequence=0,
            ),
            _make_record(
                action_type=AuditEventType.TOOL_RESPONSE,
                detail={"tool_name": "bash", "latency_ms": 10.0, "output_length": 100},
                outcome="success",
                sequence=1,
            ),
        ]
        _write_session(tmp_path, "sess-1", records)

        analyzer = TraceAnalyzer()
        sessions = analyzer.load_sessions(tmp_path)
        assert len(sessions) == 1
        assert sessions[0].tool_calls[0].arguments == {"cmd": "ls -la"}

    def test_unrecognized_action_type_ignored(self, tmp_path: Path) -> None:
        records = [
            _make_record(
                action_type=AuditEventType.TOOL_CALL,
                detail={"tool_name": "bash", "arguments": {}},
                sequence=0,
            ),
            *_make_tool_call_response_pair(
                tool_name="bash",
                outcome="success",
                seq_start=1,
            ),
        ]
        _write_session(tmp_path, "sess-1", records)

        analyzer = TraceAnalyzer()
        sessions = analyzer.load_sessions(tmp_path)
        assert len(sessions) == 1
        assert len(sessions[0].tool_calls) == 1

    def test_os_error_when_reading_file(self, tmp_path: Path) -> None:
        bad_path = tmp_path / "bad.audit.jsonl"
        bad_path.write_text('{"valid": "json", "but": "not_audit_record"}\n')
        # Make it unreadable
        import os

        os.chmod(str(bad_path), 0o000)
        try:
            analyzer = TraceAnalyzer()
            sessions = analyzer.load_sessions(tmp_path)
            assert len(sessions) == 0
        finally:
            os.chmod(str(bad_path), 0o644)

    def test_permission_event_without_tool_name_defaults(self, tmp_path: Path) -> None:
        records = [
            _make_record(
                action_type=AuditEventType.PERMISSION_CHECK,
                detail={"reason": "not in allowlist"},
                outcome="denied",
                sequence=0,
            ),
            *_make_tool_call_response_pair(seq_start=1),
        ]
        _write_session(tmp_path, "sess-1", records)

        analyzer = TraceAnalyzer()
        sessions = analyzer.load_sessions(tmp_path)
        assert len(sessions) == 1
        assert len(sessions[0].permission_denials) == 1
        assert sessions[0].permission_denials[0][0] == "unknown"

    def test_permission_grant_without_tool_name_defaults(self, tmp_path: Path) -> None:
        records = [
            _make_record(
                action_type=AuditEventType.PERMISSION_GRANT,
                detail={},
                sequence=0,
            ),
        ]
        _write_session(tmp_path, "sess-1", records)

        analyzer = TraceAnalyzer()
        sessions = analyzer.load_sessions(tmp_path)
        assert len(sessions) == 1
        assert len(sessions[0].permission_grants) == 1
        assert sessions[0].permission_grants[0] == "unknown"


class TestGenerateReportMultiSource:
    def test_report_combines_multiple_failure_sources(self, tmp_path: Path) -> None:
        records1 = [
            *_make_tool_call_response_pair(
                session_id="sess-1", tool_name="bash", outcome="error", seq_start=0
            ),
            *_make_tool_call_response_pair(
                session_id="sess-1", tool_name="file_read", outcome="timeout", seq_start=2
            ),
        ]
        records2 = [
            *_make_tool_call_response_pair(
                session_id="sess-2", tool_name="bash", outcome="error", seq_start=0
            ),
            *_make_tool_call_response_pair(
                session_id="sess-2", tool_name="bash", outcome="success", seq_start=2
            ),
        ]
        _write_session(tmp_path, "sess-1", records1)
        _write_session(tmp_path, "sess-2", records2)

        analyzer = TraceAnalyzer()
        sessions = analyzer.load_sessions(tmp_path)
        report = analyzer.generate_report(sessions)

        assert report.sessions_analyzed == 2
        bash_failure = next(f for f in report.tool_failures if f.tool_name == "bash")
        assert bash_failure.frequency == 2
        file_failure = next(f for f in report.tool_failures if f.tool_name == "file_read")
        assert file_failure.frequency == 1

    def test_report_most_used_tools_are_top_10(self, tmp_path: Path) -> None:
        records = []
        for i in range(5):
            records.extend(_make_tool_call_response_pair(tool_name=f"tool-{i}", seq_start=i * 2))
        _write_session(tmp_path, "sess-1", records)

        analyzer = TraceAnalyzer()
        sessions = analyzer.load_sessions(tmp_path)
        report = analyzer.generate_report(sessions)

        assert len(report.most_used_tools) == 5

    def test_report_no_permission_events_in_sessions(self, tmp_path: Path) -> None:
        records = [
            *_make_tool_call_response_pair(seq_start=0),
            *_make_tool_call_response_pair(tool_name="bash", seq_start=2),
        ]
        _write_session(tmp_path, "sess-1", records)

        analyzer = TraceAnalyzer()
        sessions = analyzer.load_sessions(tmp_path)
        report = analyzer.generate_report(sessions)

        assert report.permission_insights == ()

    def test_report_with_zero_latency_sessions(self, tmp_path: Path) -> None:
        records = [
            *_make_tool_call_response_pair(latency_ms=0.0, seq_start=0),
        ]
        _write_session(tmp_path, "sess-1", records)

        analyzer = TraceAnalyzer()
        sessions = analyzer.load_sessions(tmp_path)
        report = analyzer.generate_report(sessions)

        assert report.latency_stats == ()
        assert report.error_rate == 0.0


class TestErrorPatternClustering:
    def test_same_tool_different_categories(self, tmp_path: Path) -> None:
        records = [
            *_make_tool_call_response_pair(tool_name="bash", outcome="timeout", seq_start=0),
            *_make_tool_call_response_pair(
                tool_name="bash",
                arguments={"path": "not found"},
                outcome="error",
                seq_start=2,
            ),
            *_make_tool_call_response_pair(
                tool_name="bash",
                arguments={"cmd": "ls"},
                outcome="error",
                seq_start=4,
            ),
        ]
        _write_session(tmp_path, "sess-1", records)

        analyzer = TraceAnalyzer()
        sessions = analyzer.load_sessions(tmp_path)
        failures = analyzer.analyze_tool_failures(sessions)

        assert len(failures) == 3
        categories = {f.error_category for f in failures}
        assert "timeout" in categories
        assert "invalid_args" in categories
        assert "execution_error" in categories
