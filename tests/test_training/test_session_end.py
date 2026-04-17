"""Tests for ConversationLogger.log_session_end (v2.5.1).

The session_end record is the terminal signal RL pipelines use to score
a run. It must carry the exit_reason / exit_code / metrics that the audit
trail records, so both streams can be cross-referenced.
"""

from __future__ import annotations

import json
from pathlib import Path

from godspeed.training.conversation_logger import ConversationLogger


def _read_all(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_log_session_end_writes_expected_record(tmp_path: Path) -> None:
    logger = ConversationLogger(session_id="sess-1", output_dir=tmp_path)
    try:
        logger.log_system("you are helpful")
        logger.log_session_end(
            exit_reason="stopped",
            exit_code=0,
            iterations_used=3,
            tool_call_count=5,
            tool_error_count=0,
            duration_seconds=12.3456,
            cost_usd=0.012345,
            must_fix_injections=2,
        )
    finally:
        logger.close()

    records = _read_all(logger.path)
    assert len(records) == 2
    end = records[-1]
    assert end["role"] == "session_end"
    assert end["exit_reason"] == "stopped"
    assert end["exit_code"] == 0
    assert end["iterations_used"] == 3
    assert end["tool_call_count"] == 5
    assert end["tool_error_count"] == 0
    assert end["must_fix_injections"] == 2
    # Fields rounded as documented
    assert end["duration_seconds"] == 12.346
    assert end["cost_usd"] == 0.012345
    # Shared envelope fields
    assert end["session_id"] == "sess-1"
    assert "timestamp" in end


def test_multiple_session_end_records_are_append_only(tmp_path: Path) -> None:
    """The logger is append-only; a caller writing two session_end records
    (e.g., on retry in an orchestrator) should see both preserved."""
    logger = ConversationLogger(session_id="sess-2", output_dir=tmp_path)
    try:
        logger.log_session_end("timeout", 6, 50, 12, 3, 1800.0, 0.42)
        logger.log_session_end("stopped", 0, 20, 5, 0, 60.0, 0.05)
    finally:
        logger.close()

    records = _read_all(logger.path)
    assert [r["exit_reason"] for r in records] == ["timeout", "stopped"]
    assert [r["exit_code"] for r in records] == [6, 0]
