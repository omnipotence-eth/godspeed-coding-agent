"""Evolution registry — versioned history of all evolved artifacts.

Append-only JSONL storage tracking every mutation attempt, its fitness score,
safety verdict, and whether it was applied/reverted.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from godspeed.evolution.fitness import FitnessScore
from godspeed.evolution.mutator import MutationCandidate
from godspeed.evolution.safety import SafetyVerdict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclasses.dataclass(slots=True)
class EvolutionRecord:
    """A single entry in the evolution registry."""

    record_id: str
    artifact_type: str
    artifact_id: str
    original_hash: str
    mutated_hash: str
    fitness_overall: float
    fitness_confidence: float
    safety_passed: bool
    requires_review: bool
    model_used: str
    created_at: str  # ISO 8601
    applied_at: str  # ISO 8601 or "" if not applied
    reverted_at: str  # ISO 8601 or "" if not reverted

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> EvolutionRecord:
        valid = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid})


# ---------------------------------------------------------------------------
# Evolution Registry
# ---------------------------------------------------------------------------


class EvolutionRegistry:
    """Append-only JSONL registry tracking all evolution history.

    Storage layout:
        base_dir/
        ├── registry.jsonl       # append-only history
        ├── candidates/          # pending candidates
        │   └── {id}.json
        ├── applied/             # currently active mutations
        │   └── {artifact_id}.json
        └── originals/           # backups before mutation
            └── {artifact_id}.json
    """

    def __init__(self, base_dir: Path) -> None:
        self._base_dir = base_dir
        self._registry_path = base_dir / "registry.jsonl"
        self._candidates_dir = base_dir / "candidates"
        self._applied_dir = base_dir / "applied"
        self._originals_dir = base_dir / "originals"

        # Ensure directories exist
        for d in (self._candidates_dir, self._applied_dir, self._originals_dir):
            d.mkdir(parents=True, exist_ok=True)

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    def register(
        self,
        candidate: MutationCandidate,
        score: FitnessScore,
        verdict: SafetyVerdict,
    ) -> str:
        """Record a mutation attempt in the registry.

        Returns:
            The record ID.
        """
        record_id = str(uuid4())[:12]

        record = EvolutionRecord(
            record_id=record_id,
            artifact_type=candidate.artifact_type,
            artifact_id=candidate.artifact_id,
            original_hash=self._hash_text(candidate.original),
            mutated_hash=self._hash_text(candidate.mutated),
            fitness_overall=score.overall,
            fitness_confidence=score.confidence,
            safety_passed=verdict.passed,
            requires_review=verdict.requires_human_review,
            model_used=candidate.model_used,
            created_at=datetime.now(tz=UTC).isoformat(),
            applied_at="",
            reverted_at="",
        )

        # Append to registry
        self._append_record(record)

        # Save candidate details for review
        candidate_data = {
            "record_id": record_id,
            "artifact_type": candidate.artifact_type,
            "artifact_id": candidate.artifact_id,
            "original": candidate.original,
            "mutated": candidate.mutated,
            "mutation_rationale": candidate.mutation_rationale,
            "fitness": dataclasses.asdict(score),
            "safety": {
                "passed": verdict.passed,
                "checks": [
                    {"name": name, "passed": ok, "reason": reason}
                    for name, ok, reason in verdict.checks
                ],
                "requires_human_review": verdict.requires_human_review,
            },
        }
        candidate_path = self._candidates_dir / f"{record_id}.json"
        candidate_path.write_text(json.dumps(candidate_data, indent=2), encoding="utf-8")

        logger.info(
            "Registered evolution record=%s artifact=%s fitness=%.3f safety=%s",
            record_id,
            candidate.artifact_id,
            score.overall,
            verdict.passed,
        )
        return record_id

    def mark_applied(self, record_id: str, mutated_text: str) -> None:
        """Mark a record as applied and save the active mutation."""
        records = self._load_records()
        for rec in records:
            if rec.record_id == record_id:
                rec.applied_at = datetime.now(tz=UTC).isoformat()
                # Save to applied/
                applied_path = self._applied_dir / f"{rec.artifact_id}.json"
                applied_data = {
                    "record_id": record_id,
                    "artifact_id": rec.artifact_id,
                    "artifact_type": rec.artifact_type,
                    "mutated_text": mutated_text,
                    "applied_at": rec.applied_at,
                }
                applied_path.write_text(json.dumps(applied_data, indent=2), encoding="utf-8")
                break

        self._rewrite_registry(records)

    def mark_reverted(self, record_id: str) -> None:
        """Mark a record as reverted and remove from applied/."""
        records = self._load_records()
        for rec in records:
            if rec.record_id == record_id:
                rec.reverted_at = datetime.now(tz=UTC).isoformat()
                # Remove from applied/
                applied_path = self._applied_dir / f"{rec.artifact_id}.json"
                if applied_path.exists():
                    applied_path.unlink()
                break

        self._rewrite_registry(records)

    def save_original(self, artifact_id: str, original_text: str) -> None:
        """Backup original text before applying a mutation."""
        path = self._originals_dir / f"{artifact_id}.json"
        if not path.exists():  # Only save the first original
            path.write_text(
                json.dumps({"artifact_id": artifact_id, "text": original_text}, indent=2),
                encoding="utf-8",
            )

    def get_original(self, artifact_id: str) -> str | None:
        """Retrieve the backed-up original text for an artifact."""
        path = self._originals_dir / f"{artifact_id}.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("text")

    def get_applied(self, artifact_id: str) -> dict | None:
        """Get the currently applied mutation for an artifact."""
        path = self._applied_dir / f"{artifact_id}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def list_applied(self) -> list[dict]:
        """List all currently applied mutations."""
        results = []
        for path in self._applied_dir.glob("*.json"):
            results.append(json.loads(path.read_text(encoding="utf-8")))
        return results

    def get_history(self, artifact_id: str) -> list[EvolutionRecord]:
        """Get full mutation history for an artifact."""
        records = self._load_records()
        return [r for r in records if r.artifact_id == artifact_id]

    def get_record(self, record_id: str) -> EvolutionRecord | None:
        """Get a specific record by ID."""
        records = self._load_records()
        for r in records:
            if r.record_id == record_id:
                return r
        return None

    def get_candidate(self, record_id: str) -> dict | None:
        """Get full candidate details by record ID."""
        path = self._candidates_dir / f"{record_id}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def leaderboard(self, top_n: int = 10) -> list[EvolutionRecord]:
        """Get top improvements by fitness score."""
        records = self._load_records()
        # Only applied, non-reverted records
        active = [r for r in records if r.applied_at and not r.reverted_at]
        active.sort(key=lambda r: r.fitness_overall, reverse=True)
        return active[:top_n]

    def stats(self) -> dict:
        """Get summary statistics."""
        records = self._load_records()
        return {
            "total_mutations": len(records),
            "applied": sum(1 for r in records if r.applied_at),
            "reverted": sum(1 for r in records if r.reverted_at),
            "active": sum(1 for r in records if r.applied_at and not r.reverted_at),
            "safety_passed": sum(1 for r in records if r.safety_passed),
            "safety_failed": sum(1 for r in records if not r.safety_passed),
            "avg_fitness": (
                sum(r.fitness_overall for r in records) / len(records) if records else 0.0
            ),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _append_record(self, record: EvolutionRecord) -> None:
        """Append a record to the JSONL registry."""
        line = json.dumps(record.to_dict()) + "\n"
        try:
            with open(self._registry_path, "a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
        except OSError as exc:
            logger.warning("Failed to append registry record: %s", exc)
            raise

    def _load_records(self) -> list[EvolutionRecord]:
        """Load all records from the JSONL registry."""
        if not self._registry_path.exists():
            return []

        records: list[EvolutionRecord] = []
        for line in self._registry_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                records.append(EvolutionRecord.from_dict(data))
            except (json.JSONDecodeError, TypeError) as exc:
                logger.debug("Skipping malformed registry record: %s", exc)
        return records

    def _rewrite_registry(self, records: list[EvolutionRecord]) -> None:
        """Rewrite the entire registry atomically (used for updates like mark_applied).

        Writes to a temp file first, then atomically replaces the real file.
        This prevents data corruption on disk-full or mid-write crashes.
        """
        lines = [json.dumps(r.to_dict()) + "\n" for r in records]
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".jsonl", prefix="registry_", dir=self._base_dir)
        try:
            with open(tmp_fd, "w", encoding="utf-8") as f:
                f.write("".join(lines))
                f.flush()
            Path(tmp_path).replace(self._registry_path)
        except OSError as exc:
            logger.warning("Failed to rewrite registry: %s — data may be lost", exc)
            Path(tmp_path).unlink(missing_ok=True)
            raise

    @staticmethod
    def _hash_text(text: str) -> str:
        """SHA-256 hash of text content."""
        return hashlib.sha256(text.encode()).hexdigest()[:16]
