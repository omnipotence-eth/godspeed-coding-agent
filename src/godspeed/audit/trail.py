"""Hash-chained JSONL audit trail — tamper-evident logging.

Every action is recorded as a typed event. Each record includes the SHA-256
hash of the previous record, creating a verifiable chain. Secrets are
redacted before writing.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from pathlib import Path

from godspeed.audit.events import AuditEventType, AuditRecord
from godspeed.audit.redactor import redact_audit_detail

logger = logging.getLogger(__name__)


class AuditTrail:
    """Append-only, hash-chained JSONL audit log.

    Thread-safe: all writes are serialized through a lock to protect
    the hash chain (_sequence and _prev_hash) from concurrent mutations.

    One file per session: {session_id}.audit.jsonl
    """

    def __init__(self, log_dir: Path, session_id: str) -> None:
        self._log_dir = log_dir
        self._session_id = session_id
        self._prev_hash = ""
        self._record_count = 0
        self._sequence = 0
        self._lock = threading.Lock()

        # Ensure log directory exists
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = self._log_dir / f"{session_id}.audit.jsonl"

    @property
    def log_path(self) -> Path:
        return self._log_path

    @property
    def record_count(self) -> int:
        return self._record_count

    def record(
        self,
        event_type: AuditEventType | str,
        detail: dict | None = None,
        outcome: str = "success",
    ) -> AuditRecord:
        """Append a record to the audit trail.

        Thread-safe. Redacts secrets, computes hash chain, writes to JSONL.
        Uses fsync for durable writes. On I/O failure, logs error but does
        not crash the agent — the audit chain will be broken for this session.
        """
        # Redact secrets from detail
        safe_detail = redact_audit_detail(detail or {})

        with self._lock:
            # Create record with sequence number
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

            # Compute this record's hash before writing
            record_json = event.model_dump_json(exclude={"record_hash"})
            event.record_hash = hashlib.sha256(record_json.encode()).hexdigest()

            # Write with durability guarantees
            try:
                line = event.model_dump_json() + "\n"
                with open(self._log_path, "a", encoding="utf-8") as f:
                    f.write(line)
                    f.flush()
                    os.fsync(f.fileno())
            except OSError as exc:
                logger.error(
                    "Audit write failed path=%s error=%s — chain may be broken",
                    self._log_path,
                    exc,
                )
                # Still update chain state so subsequent records link correctly
                # to the record we attempted (even if it didn't persist)

            # Update chain
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

    def verify_chain(self) -> tuple[bool, str]:
        """Verify the hash chain integrity of the entire session log.

        Returns:
            (is_valid, message) tuple.
        """
        if not self._log_path.exists():
            return True, "No audit log to verify"

        prev_hash = ""
        line_num = 0

        with open(self._log_path, encoding="utf-8") as f:
            for line in f:
                line_num += 1
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    return False, f"Line {line_num}: invalid JSON"

                record = AuditRecord.model_validate(data)

                # Verify prev_hash matches
                if record.prev_hash != prev_hash:
                    return (
                        False,
                        f"Line {line_num}: prev_hash mismatch "
                        f"(expected {prev_hash[:12]}..., got {record.prev_hash[:12]}...)",
                    )

                # Verify record_hash
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
