"""E2E drift test: audit trail and conversation log must agree on session_end.

The two logs are written independently from `_headless_run`. If they ever
drift on `exit_reason`, `exit_code`, or the metrics fields, downstream RL
sees inconsistent signals. This test asserts the invariant in one
integration-style run so regressions surface immediately.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from godspeed.cli import _headless_run
from godspeed.llm.client import ChatResponse


def _read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@pytest.mark.asyncio
async def test_audit_and_conversation_log_agree_on_session_end(tmp_path: Path, monkeypatch) -> None:
    """Audit session_end and conversation session_end must carry the same
    values for exit_reason, exit_code, and every metrics field.
    """
    monkeypatch.setenv("GODSPEED_GLOBAL_DIR", str(tmp_path / "godspeed"))
    monkeypatch.setenv("GODSPEED_LOG_CONVERSATIONS", "true")

    async def fake_chat(self, messages, tools=None, task_type=None):
        return ChatResponse(content="done", finish_reason="stop")

    with (
        patch("godspeed.llm.client.LLMClient.chat", new=fake_chat),
        patch("sys.stdout.write"),
    ):
        exit_code = await _headless_run(
            task="say done",
            model="test-model",
            project_dir=tmp_path,
            auto_approve="reads",
            max_iterations=5,
            timeout=0,
            json_output=False,
        )

    audit_dir = tmp_path / "godspeed" / "audit"
    training_dir = tmp_path / "godspeed" / "training"

    audit_files = list(audit_dir.glob("*.audit.jsonl"))
    training_files = list(training_dir.glob("*.conversation.jsonl"))
    assert len(audit_files) == 1, "expected one audit log"
    assert len(training_files) == 1, "expected one conversation log"

    audit_records = _read_jsonl(audit_files[0])
    training_records = _read_jsonl(training_files[0])

    audit_end = next(r for r in audit_records if r.get("action_type") == "session_end")
    training_end = next(r for r in training_records if r.get("role") == "session_end")

    # Core invariant: both records agree on the exit signal and metrics.
    audit_detail = audit_end["action_detail"]
    for field in (
        "exit_reason",
        "exit_code",
        "iterations_used",
        "tool_call_count",
        "tool_error_count",
        "must_fix_injections",
        "duration_seconds",
        "cost_usd",
    ):
        assert audit_detail[field] == training_end[field], (
            f"Drift on field {field}: "
            f"audit={audit_detail[field]!r} vs training={training_end[field]!r}"
        )

    # Cross-check with the returned exit_code so the full chain is consistent.
    assert exit_code == audit_detail["exit_code"] == training_end["exit_code"]
