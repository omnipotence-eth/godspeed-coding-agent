"""Hash-chained JSONL audit trail — tamper-evident logging.

Every action is recorded as a typed event. Each record includes the SHA-256
hash of the previous record, creating a verifiable chain. Secrets are
redacted before writing.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import threading
from pathlib import Path

from godspeed.audit.events import AuditEventType, AuditRecord
from godspeed.audit.redactor import redact_audit_detail

logger = logging.getLogger(__name__)


class AuditWriteError(OSError):
    """Raised when an audit record cannot be durably persisted.

    A tamper-evident log is only meaningful if every event reaches disk.
    Callers must fail closed: stop executing tool calls until audit recovers.
    """


class AuditTrail:
    """Append-only, hash-chained JSONL audit log.

    Thread-safe: all writes are serialized through a lock to protect
    the hash chain (_sequence and _prev_hash) from concurrent mutations.

    One file per session: {session_id}.audit.jsonl

    Supports async batched writes for high-throughput scenarios.
    Batch writes are queued and flushed periodically or on shutdown.
    """

    def __init__(self, log_dir: Path, session_id: str, batch_size: int = 0) -> None:
        self._log_dir = log_dir
        self._session_id = session_id
        self._prev_hash = ""
        self._record_count = 0
        self._sequence = 0
        self._lock = threading.Lock()
        self._batch_size = batch_size
        self._batch_queue: list[AuditRecord] = []
        self._batch_lock = threading.Lock()

        # Ensure log directory exists
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = self._log_dir / f"{session_id}.audit.jsonl"

    @property
    def log_path(self) -> Path:
        return self._log_path

    @property
    def record_count(self) -> int:
        return self._record_count

    async def record_async(
        self,
        event_type: AuditEventType | str,
        detail: dict | None = None,
        outcome: str = "success",
    ) -> AuditRecord:
        """Async wrapper for record() — queues for batch write if batch_size > 0.

        When batch_size is set, records are queued and flushed periodically.
        Otherwise, falls back to synchronous write.
        """
        if self._batch_size > 0:
            # Async batch mode
            safe_detail = redact_audit_detail(detail or {})
            event = AuditRecord(
                session_id=self._session_id,
                sequence=self._sequence,
                action_type=AuditEventType(event_type)
                if isinstance(event_type, str)
                else event_type,
                action_detail=safe_detail,
                outcome=outcome,
                prev_hash=self._prev_hash,
            )
            record_json = event.model_dump_json(exclude={"record_hash"})
            event.record_hash = hashlib.sha256(record_json.encode()).hexdigest()

            with self._batch_lock:
                self._batch_queue.append(event)
                if len(self._batch_queue) >= self._batch_size:
                    await self._flush_batch()

            return event
        else:
            # Synchronous mode
            return self.record(event_type, detail, outcome)

    async def _flush_batch(self) -> None:
        """Flush the current batch queue to disk."""
        with self._lock:
            with self._batch_lock:
                batch_to_flush = self._batch_queue.copy()
                self._batch_queue.clear()

            if not batch_to_flush:
                return

            try:
                lines = ""
                for event in batch_to_flush:
                    lines += event.model_dump_json() + "\n"
                    self._prev_hash = event.record_hash
                    self._sequence += 1

                with open(self._log_path, "a", encoding="utf-8") as f:
                    f.write(lines)
                    f.flush()
                    os.fsync(f.fileno())

                self._record_count += len(batch_to_flush)
                logger.debug(
                    "Audit batch flushed count=%d total=%d",
                    len(batch_to_flush),
                    self._record_count,
                )
            except OSError as exc:
                logger.error(
                    "Audit batch write failed path=%s error=%s",
                    self._log_path,
                    exc,
                )
                raise AuditWriteError(f"audit batch write failed: {exc}") from exc

    async def flush_pending(self) -> None:
        """Flush any pending batched records to disk."""
        if self._batch_size > 0 and self._batch_queue:
            await self._flush_batch()

    def record(
        self,
        event_type: AuditEventType | str,
        detail: dict | None = None,
        outcome: str = "success",
    ) -> AuditRecord:
        """Append a record to the audit trail.

        Thread-safe. Redacts secrets, computes hash chain, writes to JSONL.
        Uses fsync for durable writes. On I/O failure, raises AuditWriteError
        and leaves chain state unchanged — callers must fail closed.
        """
        safe_detail = redact_audit_detail(detail or {})

        with self._lock:
            event = AuditRecord(
                session_id=self._session_id,
                sequence=self._sequence,
                action_type=AuditEventType(event_type)
                if isinstance(event_type, str)
                else event_type,
                action_detail=safe_detail,
                outcome=outcome,
                prev_hash=self._prev_hash,
            )

            record_json = event.model_dump_json(exclude={"record_hash"})
            event.record_hash = hashlib.sha256(record_json.encode()).hexdigest()

            try:
                line = event.model_dump_json() + "\n"
                with open(self._log_path, "a", encoding="utf-8") as f:
                    f.write(line)
                    f.flush()
                    os.fsync(f.fileno())
            except OSError as exc:
                logger.error(
                    "Audit write failed path=%s error=%s — refusing to advance chain",
                    self._log_path,
                    exc,
                )
                # Fail closed: do NOT advance _prev_hash / _sequence.
                # The next call will reuse the same sequence and prev_hash,
                # so a successful retry chains cleanly from the last persisted record.
                raise AuditWriteError(f"audit write failed: {exc}") from exc

            self._prev_hash = event.record_hash
            self._sequence += 1
            self._record_count += 1

        logger.debug(
            "Audit record type=%s outcome=%s seq=%d hash=%s",
            event_type,
            outcome,
            event.sequence,
            event.record_hash[:12],
        )
        return event

    def compress_session(self) -> Path | None:
        """Compress the current session's audit log to gzip.

        Rotates {session}.audit.jsonl → {session}.audit.jsonl.gz.
        The uncompressed file is removed after successful compression.

        Returns:
            Path to the compressed file, or None if nothing to compress.
        """
        if not self._log_path.exists() or self._log_path.stat().st_size == 0:
            return None

        gz_path = self._log_path.with_suffix(".jsonl.gz")
        try:
            with open(self._log_path, "rb") as f_in, gzip.open(gz_path, "wb") as f_out:
                f_out.writelines(f_in)
            self._log_path.unlink()
            logger.info(
                "Compressed audit log session=%s records=%d",
                self._session_id,
                self._record_count,
            )
            return gz_path
        except OSError as exc:
            logger.error("Audit compression failed session=%s error=%s", self._session_id, exc)
            # Clean up partial gz if it exists
            if gz_path.exists():
                gz_path.unlink(missing_ok=True)
            return None

    @staticmethod
    def _read_log_lines(log_path: Path) -> list[str]:
        """Read lines from a log file, handling both plain and gzipped formats."""
        if log_path.suffix == ".gz" or str(log_path).endswith(".jsonl.gz"):
            with gzip.open(log_path, "rt", encoding="utf-8") as f:
                return f.readlines()
        with open(log_path, encoding="utf-8") as f:
            return f.readlines()

    def verify_chain(self, log_path: Path | None = None) -> tuple[bool, str]:
        """Verify the hash chain integrity of a session log.

        Supports both plain .jsonl and compressed .jsonl.gz files.

        Args:
            log_path: Path to verify. Defaults to current session's log.
                      Checks both uncompressed and compressed paths.

        Returns:
            (is_valid, message) tuple.
        """
        target = log_path or self._log_path
        # If the uncompressed file doesn't exist, try the compressed version
        if not target.exists():
            gz_target = target.with_suffix(".jsonl.gz")
            if gz_target.exists():
                target = gz_target
            else:
                return True, "No audit log to verify"

        try:
            lines = self._read_log_lines(target)
        except (OSError, gzip.BadGzipFile) as exc:
            return False, f"Cannot read log: {exc}"

        prev_hash = ""
        line_num = 0

        for raw_line in lines:
            line_num += 1
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            try:
                data = json.loads(raw_line)
            except json.JSONDecodeError:
                return False, f"Line {line_num}: invalid JSON"

            record = AuditRecord.model_validate(data)

            if record.prev_hash != prev_hash:
                return (
                    False,
                    f"Line {line_num}: prev_hash mismatch "
                    f"(expected {prev_hash[:12]}..., got {record.prev_hash[:12]}...)",
                )

            record_json = record.model_copy(update={"record_hash": ""}).model_dump_json(
                exclude={"record_hash"}
            )
            expected_hash = hashlib.sha256(record_json.encode()).hexdigest()

            if record.record_hash != expected_hash:
                return (
                    False,
                    f"Line {line_num}: record_hash mismatch "
                    f"(expected {expected_hash[:12]}..., got {record.record_hash[:12]}...)",
                )

            prev_hash = record.record_hash

        return True, f"Chain verified: {line_num} records"

    def cleanup_expired(self, retention_days: int = 30) -> int:
        """Remove audit logs older than retention_days.

        Scans the log directory for session files (both plain and compressed)
        and deletes those whose last modification time exceeds the retention
        period. Skips the current session's log file.

        Returns:
            Number of expired log files removed.
        """
        import time

        if retention_days <= 0:
            return 0

        cutoff = time.time() - (retention_days * 86400)
        removed = 0

        # Check both .jsonl and .jsonl.gz files
        patterns = ["*.audit.jsonl", "*.audit.jsonl.gz"]
        for pattern in patterns:
            for log_file in self._log_dir.glob(pattern):
                # Never delete the current session's log (compressed or not)
                if (
                    log_file == self._log_path
                    or log_file.stem.replace(".audit", "") == self._session_id
                ):
                    continue
                try:
                    if log_file.stat().st_mtime < cutoff:
                        log_file.unlink()
                        removed += 1
                        logger.info("Removed expired audit log path=%s", log_file.name)
                except OSError as exc:
                    logger.warning(
                        "Failed to remove expired audit log path=%s error=%s", log_file, exc
                    )

        if removed:
            logger.info("Audit cleanup removed=%d retention_days=%d", removed, retention_days)
        return removed
