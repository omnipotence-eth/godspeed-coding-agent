"""Tests for hash-chained audit trail."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from godspeed.audit.events import AuditEventType
from godspeed.audit.trail import AuditTrail, AuditWriteError


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

    def test_cleanup_removes_compressed_logs(self, audit_dir: Path) -> None:
        """Expired compressed logs are also cleaned up."""
        import gzip
        import os
        import time

        trail = AuditTrail(log_dir=audit_dir, session_id="current-session")
        trail.record(AuditEventType.SESSION_START)

        # Create an old compressed log
        old_gz = audit_dir / "old-session.audit.jsonl.gz"
        with gzip.open(old_gz, "wt", encoding="utf-8") as f:
            f.write('{"fake": "record"}\n')
        old_time = time.time() - (60 * 86400)
        os.utime(old_gz, (old_time, old_time))

        removed = trail.cleanup_expired(retention_days=30)
        assert removed == 1
        assert not old_gz.exists()


class TestAuditCompression:
    """Test audit log compression."""

    def test_compress_creates_gzip(self, trail: AuditTrail) -> None:
        trail.record(AuditEventType.SESSION_START, {"model": "test"})
        trail.record(AuditEventType.TOOL_CALL, {"tool": "shell"})
        trail.record(AuditEventType.SESSION_END)

        gz_path = trail.compress_session()
        assert gz_path is not None
        assert gz_path.exists()
        assert str(gz_path).endswith(".jsonl.gz")
        # Original removed
        assert not trail.log_path.exists()

    def test_compressed_log_verifiable(self, trail: AuditTrail) -> None:
        trail.record(AuditEventType.SESSION_START, {"model": "test"})
        trail.record(AuditEventType.TOOL_CALL, {"tool": "file_read"})
        trail.record(AuditEventType.SESSION_END)

        gz_path = trail.compress_session()
        assert gz_path is not None

        # Verify the compressed log
        is_valid, msg = trail.verify_chain(gz_path)
        assert is_valid, msg
        assert "3 records" in msg


class TestAuditThreadSafety:
    """Adversarial: sync record() must be safe across multiple threads."""

    def test_concurrent_records_preserve_chain(self, audit_dir: Path) -> None:
        import concurrent.futures
        import threading

        trail = AuditTrail(log_dir=audit_dir, session_id="thread-test")
        barrier = threading.Barrier(10)

        def _record_batch(_trail: AuditTrail) -> None:
            barrier.wait()
            for _ in range(10):
                _trail.record(AuditEventType.TOOL_CALL, {"tool": "shell"})

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            list(executor.map(_record_batch, [trail] * 10))

        assert trail.record_count == 100
        is_valid, msg = trail.verify_chain()
        assert is_valid, msg


class TestAuditFailClosed:
    """Adversarial: audit must fail closed when writes cannot persist.

    A tamper-evident log is worthless if the agent keeps running when it
    cannot record. Every write failure must surface to the caller, and the
    in-memory chain state must not advance past unpersisted records.
    """

    def test_write_failure_raises(self, trail: AuditTrail) -> None:
        """OSError during write must propagate as AuditWriteError."""
        with (
            patch("builtins.open", side_effect=OSError("disk full")),
            pytest.raises(AuditWriteError, match="disk full"),
        ):
            trail.record(AuditEventType.TOOL_CALL, {"tool": "shell"})

    def test_write_failure_preserves_chain_state(self, trail: AuditTrail) -> None:
        """After a write failure, chain state is unchanged — recoverable."""
        first = trail.record(AuditEventType.SESSION_START)
        prev_count = trail.record_count
        prev_hash = first.record_hash

        # Close handle so next record() will re-open and hit the mock
        trail.close()

        with (
            patch("builtins.open", side_effect=OSError("disk full")),
            pytest.raises(AuditWriteError),
        ):
            trail.record(AuditEventType.TOOL_CALL, {"tool": "shell"})

        # State did NOT advance
        assert trail.record_count == prev_count
        # Next successful record should chain to the last persisted one
        second = trail.record(AuditEventType.TOOL_CALL, {"tool": "file_read"})
        assert second.prev_hash == prev_hash

    def test_chain_verifies_after_recovery(self, trail: AuditTrail) -> None:
        """Full chain stays verifiable across a failed + recovered write."""
        trail.record(AuditEventType.SESSION_START)

        # Close handle so next record() will re-open and hit the mock
        trail.close()

        with (
            patch("builtins.open", side_effect=OSError("transient")),
            pytest.raises(AuditWriteError),
        ):
            trail.record(AuditEventType.TOOL_CALL, {"tool": "shell"})

        trail.record(AuditEventType.TOOL_CALL, {"tool": "file_read"})
        trail.record(AuditEventType.SESSION_END)

        is_valid, msg = trail.verify_chain()
        assert is_valid, msg
        assert "3 records" in msg  # failed one was never persisted

    def test_fsync_failure_also_raises(self, trail: AuditTrail) -> None:
        """fsync failures are as fatal as write failures."""
        with (
            patch("os.fsync", side_effect=OSError("fsync failed")),
            pytest.raises(AuditWriteError, match="fsync failed"),
        ):
            trail.record(AuditEventType.TOOL_CALL, {"tool": "shell"})

    def test_compress_empty_log_returns_none(self, audit_dir: Path) -> None:
        trail = AuditTrail(log_dir=audit_dir, session_id="empty-session")
        result = trail.compress_session()
        assert result is None

    def test_compress_idempotent(self, trail: AuditTrail) -> None:
        trail.record(AuditEventType.SESSION_START)
        trail.compress_session()
        # Second compress — original is gone, should return None
        result = trail.compress_session()
        assert result is None

    def test_verify_chain_finds_compressed(self, trail: AuditTrail) -> None:
        """verify_chain() transparently finds .gz when .jsonl is missing."""
        trail.record(AuditEventType.SESSION_START, {"model": "test"})
        trail.record(AuditEventType.SESSION_END)
        trail.compress_session()

        # verify_chain with default path should find the .gz
        is_valid, msg = trail.verify_chain()
        assert is_valid, msg
