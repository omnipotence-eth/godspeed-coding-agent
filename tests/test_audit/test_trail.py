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

    def test_prev_hash_mismatch_detected(self, trail: AuditTrail) -> None:
        import json

        trail.record(AuditEventType.SESSION_START)
        trail.record(AuditEventType.TOOL_CALL, {"tool": "shell"})
        trail.record(AuditEventType.SESSION_END)

        lines = trail.log_path.read_text().splitlines()
        records = [json.loads(line) for line in lines if line.strip()]
        records[2]["prev_hash"] = "0000000000000000000000000000000000000000000000000000000000000000"
        new_content = "\n".join(json.dumps(r) for r in records) + "\n"
        trail.log_path.write_text(new_content)

        is_valid, msg = trail.verify_chain()
        assert not is_valid
        assert "prev_hash" in msg.lower()

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


class TestAuditAsyncSafety:
    """Adversarial: async arecord() must handle write failures correctly."""

    @pytest.mark.asyncio
    async def test_arecord_write_failure_raises(self, trail: AuditTrail) -> None:
        with (
            patch("builtins.open", side_effect=OSError("disk full")),
            pytest.raises(AuditWriteError, match="disk full"),
        ):
            await trail.arecord(AuditEventType.TOOL_CALL, {"tool": "shell"})

    @pytest.mark.asyncio
    async def test_arecord_preserves_chain_on_failure(self, trail: AuditTrail) -> None:
        first = trail.record(AuditEventType.SESSION_START)
        prev_count = trail.record_count
        prev_hash = first.record_hash

        trail.close()

        with (
            patch("builtins.open", side_effect=OSError("disk full")),
            pytest.raises(AuditWriteError),
        ):
            await trail.arecord(AuditEventType.TOOL_CALL, {"tool": "shell"})

        assert trail.record_count == prev_count
        second = trail.record(AuditEventType.TOOL_CALL, {"tool": "file_read"})
        assert second.prev_hash == prev_hash

    @pytest.mark.asyncio
    async def test_arecord_writes_successfully(self, trail: AuditTrail) -> None:
        record = await trail.arecord(AuditEventType.SESSION_START, {"model": "test"})
        assert record.action_type == AuditEventType.SESSION_START
        assert trail.record_count == 1
        assert trail.log_path.exists()


class TestAuditClose:
    """Tests for close(), aclose(), and context manager behavior."""

    def test_close_with_file_open(self, trail: AuditTrail) -> None:
        trail.record(AuditEventType.SESSION_START)
        assert trail._file is not None
        trail.close()
        assert trail._file is None

    def test_close_when_already_closed(self, trail: AuditTrail) -> None:
        trail.close()
        trail.close()

    def test_close_fsync_failure_logged(self, trail: AuditTrail) -> None:
        trail.record(AuditEventType.SESSION_START)
        with patch("os.fsync", side_effect=OSError("fsync fail")):
            trail.close()
        assert trail._file is None

    def test_close_after_nothing_written(self, audit_dir: Path) -> None:
        t = AuditTrail(log_dir=audit_dir, session_id="empty-close")
        t.close()

    @pytest.mark.asyncio
    async def test_aclose_with_file_open(self, trail: AuditTrail) -> None:
        trail.record(AuditEventType.SESSION_START)
        await trail.aclose()
        assert trail._file is None

    @pytest.mark.asyncio
    async def test_aclose_when_already_closed(self, trail: AuditTrail) -> None:
        await trail.aclose()

    @pytest.mark.asyncio
    async def test_aclose_fsync_failure_logged(self, trail: AuditTrail) -> None:
        trail.record(AuditEventType.SESSION_START)
        with patch("os.fsync", side_effect=OSError("fsync fail")):
            await trail.aclose()
        assert trail._file is None

    def test_context_manager(self, audit_dir: Path) -> None:
        with AuditTrail(log_dir=audit_dir, session_id="ctx-test") as t:
            t.record(AuditEventType.SESSION_START)
            assert t.log_path.exists()
        assert t._file is None

    def test_context_manager_exit_on_exception(self, audit_dir: Path) -> None:
        t = AuditTrail(log_dir=audit_dir, session_id="ctx-exc")
        try:
            with t:
                t.record(AuditEventType.SESSION_START)
                raise ValueError("test exception")
        except ValueError:
            pass
        assert t._file is None

    def test_del_cleans_up_file_handle(self, audit_dir: Path) -> None:
        import gc

        t = AuditTrail(log_dir=audit_dir, session_id="del-test")
        t.record(AuditEventType.SESSION_START)
        assert t._file is not None
        fh = t._file
        del t
        gc.collect()
        assert fh.closed


class TestAuditCompressionEdgeCases:
    """Edges for compression and verification."""

    def test_compress_file_write_failure(self, trail: AuditTrail) -> None:
        trail.record(AuditEventType.SESSION_START)
        with patch("gzip.open", side_effect=OSError("no space")):
            result = trail.compress_session()
        assert result is None
        assert trail.log_path.exists()

    def test_compress_write_midstream_failure(self, trail: AuditTrail) -> None:
        import gzip

        trail.record(AuditEventType.SESSION_START)
        trail.record(AuditEventType.TOOL_CALL)
        gz_path = trail.log_path.with_suffix(".jsonl.gz")

        _discard = open
        _discard = gzip.open

        def _failing_writelines(self, src):
            raise OSError("midstream write error")

        with patch("gzip.GzipFile.writelines", _failing_writelines):
            result = trail.compress_session()
            assert result is None
            assert not gz_path.exists()

    def test_compress_with_gz_partial_cleanup(self, trail: AuditTrail) -> None:

        trail.record(AuditEventType.SESSION_START)
        _discard = trail.log_path.with_suffix(".jsonl.gz")

        with (
            patch("gzip.open", side_effect=OSError("write error")),
            patch("pathlib.Path.unlink") as mock_unlink,
        ):
            result = trail.compress_session()
            assert result is None

    def test_verify_chain_cannot_read_file(self, trail: AuditTrail) -> None:
        trail.record(AuditEventType.SESSION_START)
        with patch.object(trail, "_read_log_lines", side_effect=OSError("perm denied")):
            is_valid, msg = trail.verify_chain()
            assert not is_valid
            assert "Cannot read log" in msg

    def test_verify_chain_bad_gzip(self, audit_dir: Path) -> None:

        trail = AuditTrail(log_dir=audit_dir, session_id="bad-gz")
        trail.record(AuditEventType.SESSION_START)
        gz_path = trail.compress_session()
        assert gz_path is not None

        with open(gz_path, "wb") as f:
            f.write(b"this is not a valid gzip file")

        is_valid, msg = trail.verify_chain(gz_path)
        assert not is_valid

    def test_verify_chain_with_blank_lines(self, trail: AuditTrail) -> None:
        trail.record(AuditEventType.SESSION_START)
        trail.record(AuditEventType.SESSION_END)

        lines = trail.log_path.read_text().splitlines(True)
        trail.log_path.write_text(lines[0] + "\n\n" + lines[1] + "\n")

        is_valid, msg = trail.verify_chain()
        assert is_valid, msg
        assert "Chain verified" in msg

    def test_verify_chain_with_invalid_json_line(self, trail: AuditTrail) -> None:
        trail.record(AuditEventType.SESSION_START)
        trail.record(AuditEventType.TOOL_CALL, {"tool": "shell"})
        trail.record(AuditEventType.SESSION_END)

        lines = trail.log_path.read_text().splitlines(True)
        corrupted = lines[0] + "this is not json\n" + lines[2]
        trail.log_path.write_text(corrupted)

        is_valid, msg = trail.verify_chain()
        assert not is_valid
        assert "invalid JSON" in msg.lower() or "invalid json" in msg.lower()

    def test_verify_chain_no_log_file(self, audit_dir: Path) -> None:
        trail = AuditTrail(log_dir=audit_dir, session_id="no-log")
        is_valid, msg = trail.verify_chain()
        assert is_valid
        assert "No audit log" in msg

    def test_verify_chain_with_explicit_path(self, trail: AuditTrail) -> None:
        trail.record(AuditEventType.SESSION_START)
        trail.record(AuditEventType.SESSION_END)
        is_valid, msg = trail.verify_chain(trail.log_path)
        assert is_valid

    def test_compress_empty_file_returns_none(self, audit_dir: Path) -> None:
        trail = AuditTrail(log_dir=audit_dir, session_id="empty-file")
        trail.log_path.write_text("")
        result = trail.compress_session()
        assert result is None


class TestAuditRetentionEdgeCases:
    def test_cleanup_failed_unlink_logged(self, audit_dir: Path) -> None:
        import os
        import time

        trail = AuditTrail(log_dir=audit_dir, session_id="current-session")
        trail.record(AuditEventType.SESSION_START)

        old_log = audit_dir / "old-session.audit.jsonl"
        old_log.write_text('{"fake": "record"}\n')
        old_time = time.time() - (60 * 86400)
        os.utime(old_log, (old_time, old_time))

        with patch("pathlib.Path.unlink", side_effect=OSError("permission denied")):
            removed = trail.cleanup_expired(retention_days=30)
            assert removed == 0
            assert old_log.exists()

    def test_cleanup_skips_current_stem_match(self, audit_dir: Path) -> None:
        trail = AuditTrail(log_dir=audit_dir, session_id="my-session")
        trail.record(AuditEventType.SESSION_START)

        sibling = audit_dir / "my-session.audit.jsonl.gz"
        sibling.write_text("")

        removed = trail.cleanup_expired(retention_days=1)
        assert removed == 0
        assert sibling.exists()


# ---------------------------------------------------------------------------
# Test: verify_chain edge cases — corrupt gzip, blank lines, etc.
# ---------------------------------------------------------------------------


class TestVerifyChainExpanded:
    def test_verify_chain_corrupt_gzip_file(self, audit_dir: Path) -> None:

        trail = AuditTrail(log_dir=audit_dir, session_id="corrupt-gz")
        trail.record(AuditEventType.SESSION_START, {"model": "test"})
        gz_path = trail.compress_session()
        assert gz_path is not None

        with open(gz_path, "wb") as f:
            f.write(b"this is not valid gzip at all")

        is_valid, msg = trail.verify_chain(gz_path)
        assert not is_valid
        assert "Cannot read log" in msg

    def test_verify_chain_with_blank_lines_only(self, audit_dir: Path) -> None:
        trail = AuditTrail(log_dir=audit_dir, session_id="blank-lines")
        trail.log_path.write_text("\n\n   \n\n")
        is_valid, msg = trail.verify_chain()
        assert is_valid
        assert "Chain verified" in msg

    def test_verify_chain_with_invalid_json_mid(self, audit_dir: Path) -> None:
        trail = AuditTrail(log_dir=audit_dir, session_id="bad-json")
        trail.record(AuditEventType.SESSION_START, {"model": "test"})
        trail.record(AuditEventType.TOOL_CALL, {"tool": "shell"})
        trail.record(AuditEventType.SESSION_END)

        lines = trail.log_path.read_text().splitlines(True)
        corrupted = lines[0] + "{{{ broken json\n" + lines[1] + lines[2]
        trail.log_path.write_text(corrupted)

        is_valid, msg = trail.verify_chain()
        assert not is_valid
        assert "invalid JSON" in msg.lower() or "invalid json" in msg.lower()

    def test_verify_chain_missing_log_file_no_gz(self, audit_dir: Path) -> None:
        trail = AuditTrail(log_dir=audit_dir, session_id="no-files")
        is_valid, msg = trail.verify_chain()
        assert is_valid
        assert "No audit log" in msg

    def test_verify_chain_with_explicit_gz_path(self, audit_dir: Path) -> None:
        trail = AuditTrail(log_dir=audit_dir, session_id="explicit-gz")
        trail.record(AuditEventType.SESSION_START, {"model": "test"})
        trail.record(AuditEventType.SESSION_END)
        gz_path = trail.compress_session()
        assert gz_path is not None

        is_valid, msg = trail.verify_chain(gz_path)
        assert is_valid, msg
        assert "Chain verified" in msg

    def test_verify_chain_non_existent_path(self, audit_dir: Path) -> None:
        trail = AuditTrail(log_dir=audit_dir, session_id="any")
        fake_path = audit_dir / "does-not-exist.audit.jsonl"
        is_valid, msg = trail.verify_chain(fake_path)
        assert is_valid
        assert "No audit log" in msg


# ---------------------------------------------------------------------------
# Test: compression edge cases — midstream failure, partial cleanup
# ---------------------------------------------------------------------------


class TestCompressionExpanded:
    def test_compress_midstream_unlink_keeps_original(self, trail: AuditTrail) -> None:
        trail.record(AuditEventType.SESSION_START, {"model": "test"})
        trail.record(AuditEventType.TOOL_CALL, {"tool": "shell"})

        open_count = 0
        real_open = open

        def _failing_open(*args, **kwargs):
            nonlocal open_count
            open_count += 1
            if open_count == 1:
                return real_open(*args, **kwargs)
            raise OSError("write failure")

        _discard = trail.log_path.with_suffix(".jsonl.gz")
        with patch("gzip.open", side_effect=OSError("write failure")):
            result = trail.compress_session()

        assert result is None
        assert trail.log_path.exists()

    def test_compress_already_compressed(self, trail: AuditTrail) -> None:
        trail.record(AuditEventType.SESSION_START)
        trail.compress_session()
        result = trail.compress_session()
        assert result is None

    def test_compress_creates_parents(self, audit_dir: Path) -> None:
        trail = AuditTrail(log_dir=audit_dir / "nested" / "logs", session_id="deep")
        trail.record(AuditEventType.SESSION_START, {"model": "test"})
        result = trail.compress_session()
        assert result is not None
        assert result.exists()


# ---------------------------------------------------------------------------
# Test: session cleanup edge cases — unlink failure, non-matching stem
# ---------------------------------------------------------------------------


class TestCleanupExpanded:
    def test_cleanup_unlink_failure_still_counts_others(self, audit_dir: Path) -> None:
        import os
        import time

        trail = AuditTrail(log_dir=audit_dir, session_id="current")
        trail.record(AuditEventType.SESSION_START)

        old1 = audit_dir / "old1.audit.jsonl"
        old1.write_text('{"fake": "record"}\n')
        old_time = time.time() - (60 * 86400)
        os.utime(old1, (old_time, old_time))

        old2 = audit_dir / "old2.audit.jsonl"
        old2.write_text('{"fake": "record"}\n')
        os.utime(old2, (old_time, old_time))

        with patch("pathlib.Path.unlink", side_effect=OSError("permission denied")):
            removed = trail.cleanup_expired(retention_days=30)
            assert removed == 0

    def test_cleanup_non_matching_stem_kept(self, audit_dir: Path) -> None:
        trail = AuditTrail(log_dir=audit_dir, session_id="my-session")
        trail.record(AuditEventType.SESSION_START)

        unrelated = audit_dir / "my-session-v2.audit.jsonl"
        unrelated.write_text('{"test": true}\n')

        removed = trail.cleanup_expired(retention_days=1)
        assert removed == 0
        assert unrelated.exists()

    def test_cleanup_compressed_non_matching_stem(self, audit_dir: Path) -> None:
        import gzip
        import os
        import time

        trail = AuditTrail(log_dir=audit_dir, session_id="session-a")
        trail.record(AuditEventType.SESSION_START)

        old_gz = audit_dir / "other-session.audit.jsonl.gz"
        with gzip.open(old_gz, "wt", encoding="utf-8") as f:
            f.write('{"fake": "record"}\n')
        old_time = time.time() - (90 * 86400)
        os.utime(old_gz, (old_time, old_time))

        removed = trail.cleanup_expired(retention_days=30)
        assert removed == 1
        assert not old_gz.exists()

    def test_cleanup_negative_retention_noop(self, audit_dir: Path) -> None:
        trail = AuditTrail(log_dir=audit_dir, session_id="test")
        removed = trail.cleanup_expired(retention_days=-5)
        assert removed == 0


# ---------------------------------------------------------------------------
# Test: context manager enter/exit and del scenarios
# ---------------------------------------------------------------------------


class TestContextManagerExpanded:
    def test_enter_returns_self(self, audit_dir: Path) -> None:
        trail = AuditTrail(log_dir=audit_dir, session_id="ctx-test")
        assert trail.__enter__() is trail
        trail.close()

    def test_exit_with_no_exception(self, audit_dir: Path) -> None:
        trail = AuditTrail(log_dir=audit_dir, session_id="ctx-test")
        trail.record(AuditEventType.SESSION_START)
        trail.__exit__(None, None, None)
        assert trail._file is None

    def test_exit_with_exception_still_closes(self, audit_dir: Path) -> None:
        trail = AuditTrail(log_dir=audit_dir, session_id="ctx-test")
        trail.record(AuditEventType.SESSION_START)
        trail.__exit__(ValueError, ValueError("bad"), None)
        assert trail._file is None

    def test_del_without_explicit_close(self, audit_dir: Path) -> None:
        trail = AuditTrail(log_dir=audit_dir, session_id="gc-test")
        trail.record(AuditEventType.SESSION_START)
        fh = trail._file
        assert fh is not None
        trail = None  # bypass explicit delete
        assert trail._file is None
        assert fh.closed

    def test_del_when_already_closed(self, audit_dir: Path) -> None:
        trail = AuditTrail(log_dir=audit_dir, session_id="gc-test")
        trail = None  # bypass explicit delete

    def test_del_suppresses_errors(self, audit_dir: Path) -> None:
        trail = AuditTrail(log_dir=audit_dir, session_id="gc-test")
        trail.record(AuditEventType.SESSION_START)
        with patch.object(trail._file, "close", side_effect=OSError("close error")):
            trail = None  # bypass explicit delete
        assert trail._file is None


# ---------------------------------------------------------------------------
# Test: write failure during aclose preserves chain
# ---------------------------------------------------------------------------


class TestAsyncRecordFailureExpanded:
    @pytest.mark.asyncio
    async def test_arecord_seq_increments_only_on_success(self, trail: AuditTrail) -> None:
        assert trail.record_count == 0
        rec = await trail.arecord(AuditEventType.SESSION_START, {"model": "test"})
        assert trail.record_count == 1
        assert rec.sequence == 0

    @pytest.mark.asyncio
    async def test_arecord_chains_to_previous_sync_record(self, trail: AuditTrail) -> None:
        first = trail.record(AuditEventType.SESSION_START)
        second = await trail.arecord(AuditEventType.TOOL_CALL, {"tool": "x"})
        assert second.prev_hash == first.record_hash


# ---------------------------------------------------------------------------
# Test: record with str event type
# ---------------------------------------------------------------------------


class TestRecordWithStrEventType:
    def test_str_event_type_converted_to_enum(self, trail: AuditTrail) -> None:
        rec = trail.record("tool_call", {"tool": "file_read"})
        assert rec.action_type == AuditEventType.TOOL_CALL


class TestArecordWithStrEventType:
    @pytest.mark.asyncio
    async def test_async_str_event_type(self, trail: AuditTrail) -> None:
        rec = await trail.arecord("tool_call", {"tool": "file_read"})
        assert rec.action_type == AuditEventType.TOOL_CALL


# ---------------------------------------------------------------------------
# Test: write failure midstream during event logging
# ---------------------------------------------------------------------------


class TestWriteFailureMidStream:
    def test_fsync_interval_triggers_fsync(self, trail: AuditTrail) -> None:
        import os

        original_fsync = os.fsync
        fsync_calls = []

        def _counting_fsync(fd):
            fsync_calls.append(1)
            return original_fsync(fd)

        with patch("os.fsync", side_effect=_counting_fsync):
            for i in range(15):
                trail.record(AuditEventType.TOOL_CALL, {"tool": f"tool-{i}"})

        assert len(fsync_calls) >= 1

    def test_write_failure_mid_stream_is_fatal(self, trail: AuditTrail) -> None:
        first = trail.record(AuditEventType.SESSION_START)
        prev_hash = first.record_hash
        prev_count = trail.record_count

        with patch.object(trail, "_write_record", side_effect=AuditWriteError("disk gone")):
            with pytest.raises(AuditWriteError):
                trail.record(AuditEventType.TOOL_CALL, {"tool": "x"})

        assert trail.record_count == prev_count
        assert trail._prev_hash == prev_hash


class TestLogPathProperty:
    def test_log_path_property(self, audit_dir: Path) -> None:
        trail = AuditTrail(log_dir=audit_dir, session_id="test")
        assert trail.log_path == audit_dir / "test.audit.jsonl"


class TestRecordDetailEdgeCases:
    def test_none_detail_handled(self, trail: AuditTrail) -> None:
        rec = trail.record(AuditEventType.SESSION_START)
        assert rec.action_detail == {}

    def test_empty_dict_detail(self, trail: AuditTrail) -> None:
        rec = trail.record(AuditEventType.TOOL_CALL, {})
        assert rec.action_detail == {}
