"""Tests for hash-chained audit trail."""

from __future__ import annotations

from pathlib import Path

import pytest

from godspeed.audit.events import AuditEventType
from godspeed.audit.trail import AuditTrail


@pytest.fixture
def audit_dir(tmp_path: Path) -> Path:
    d = tmp_path / "audit"
    d.mkdir()
    return d


@pytest.fixture
def trail(audit_dir: Path) -> AuditTrail:
    return AuditTrail(log_dir=audit_dir, session_id="test-session")


class TestAuditTrail:
    """Test audit trail recording and verification."""

    def test_record_creates_file(self, trail: AuditTrail) -> None:
        trail.record(AuditEventType.SESSION_START, {"model": "test"})
        assert trail.log_path.exists()

    def test_record_increments_count(self, trail: AuditTrail) -> None:
        assert trail.record_count == 0
        trail.record(AuditEventType.SESSION_START)
        assert trail.record_count == 1
        trail.record(AuditEventType.TOOL_CALL, {"tool": "file_read"})
        assert trail.record_count == 2

    def test_record_returns_audit_record(self, trail: AuditTrail) -> None:
        record = trail.record(AuditEventType.TOOL_CALL, {"tool": "shell"})
        assert record.session_id == "test-session"
        assert record.action_type == AuditEventType.TOOL_CALL
        assert record.record_hash != ""

    def test_first_record_has_empty_prev_hash(self, trail: AuditTrail) -> None:
        record = trail.record(AuditEventType.SESSION_START)
        assert record.prev_hash == ""

    def test_second_record_chains_to_first(self, trail: AuditTrail) -> None:
        first = trail.record(AuditEventType.SESSION_START)
        second = trail.record(AuditEventType.TOOL_CALL)
        assert second.prev_hash == first.record_hash
        assert second.prev_hash != ""

    def test_chain_verification_passes(self, trail: AuditTrail) -> None:
        trail.record(AuditEventType.SESSION_START, {"model": "test"})
        trail.record(AuditEventType.TOOL_CALL, {"tool": "file_read"})
        trail.record(AuditEventType.TOOL_CALL, {"tool": "shell"})
        trail.record(AuditEventType.SESSION_END)

        is_valid, msg = trail.verify_chain()
        assert is_valid, msg
        assert "4 records" in msg

    def test_tampered_record_detected(self, trail: AuditTrail) -> None:
        trail.record(AuditEventType.SESSION_START)
        trail.record(AuditEventType.TOOL_CALL, {"tool": "shell", "command": "ls"})
        trail.record(AuditEventType.SESSION_END)

        # Tamper with the log file — change a value in the middle record
        content = trail.log_path.read_text()
        tampered = content.replace('"ls"', '"rm -rf /"')
        trail.log_path.write_text(tampered)

        is_valid, msg = trail.verify_chain()
        assert not is_valid
        assert "mismatch" in msg.lower()

    def test_empty_log_verifies(self, trail: AuditTrail) -> None:
        is_valid, _msg = trail.verify_chain()
        assert is_valid


class TestAuditRedaction:
    """Test that secrets are redacted in audit logs."""

    def test_api_key_redacted_in_detail(self, trail: AuditTrail) -> None:
        trail.record(
            AuditEventType.TOOL_CALL,
            {"command": "export API_KEY='sk-proj-1234567890abcdefghijklmnopqrstuv'"},
        )
        content = trail.log_path.read_text()
        assert "sk-proj" not in content
        assert "REDACTED" in content

    def test_password_redacted_in_detail(self, trail: AuditTrail) -> None:
        trail.record(
            AuditEventType.TOOL_CALL,
            {"output": "password = 'my_secret_password_here'"},
        )
        content = trail.log_path.read_text()
        assert "my_secret_password_here" not in content

    def test_safe_content_preserved(self, trail: AuditTrail) -> None:
        trail.record(
            AuditEventType.TOOL_CALL,
            {"tool": "file_read", "path": "README.md"},
        )
        content = trail.log_path.read_text()
        assert "README.md" in content


class TestAuditRetention:
    """Test audit log retention cleanup."""

    def test_cleanup_removes_old_logs(self, audit_dir: Path) -> None:
        """Expired session logs are removed by cleanup."""
        import os
        import time

        # Create a "current" trail
        trail = AuditTrail(log_dir=audit_dir, session_id="current-session")
        trail.record(AuditEventType.SESSION_START)

        # Create an "old" log file manually
        old_log = audit_dir / "old-session.audit.jsonl"
        old_log.write_text('{"fake": "record"}\n')
        # Backdate modification time to 60 days ago
        old_time = time.time() - (60 * 86400)
        os.utime(old_log, (old_time, old_time))

        removed = trail.cleanup_expired(retention_days=30)
        assert removed == 1
        assert not old_log.exists()
        assert trail.log_path.exists()  # current session untouched

    def test_cleanup_preserves_recent_logs(self, audit_dir: Path) -> None:
        """Recent session logs are preserved by cleanup."""
        trail = AuditTrail(log_dir=audit_dir, session_id="current-session")
        trail.record(AuditEventType.SESSION_START)

        # Create a "recent" log file (created just now)
        recent_log = audit_dir / "recent-session.audit.jsonl"
        recent_log.write_text('{"fake": "record"}\n')

        removed = trail.cleanup_expired(retention_days=30)
        assert removed == 0
        assert recent_log.exists()

    def test_cleanup_never_deletes_current_session(self, audit_dir: Path) -> None:
        """Current session log is never deleted even if old."""
        import os
        import time

        trail = AuditTrail(log_dir=audit_dir, session_id="current-session")
        trail.record(AuditEventType.SESSION_START)

        # Backdate the current session's log
        old_time = time.time() - (90 * 86400)
        os.utime(trail.log_path, (old_time, old_time))

        removed = trail.cleanup_expired(retention_days=30)
        assert removed == 0
        assert trail.log_path.exists()

    def test_cleanup_with_zero_retention_is_noop(self, audit_dir: Path) -> None:
        """retention_days=0 disables cleanup."""
        trail = AuditTrail(log_dir=audit_dir, session_id="test")
        removed = trail.cleanup_expired(retention_days=0)
        assert removed == 0
