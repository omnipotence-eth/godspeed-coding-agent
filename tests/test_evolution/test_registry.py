"""Tests for the evolution registry — versioned history of evolved artifacts."""

from __future__ import annotations

from pathlib import Path

import pytest

from godspeed.evolution.fitness import FitnessScore
from godspeed.evolution.mutator import MutationCandidate
from godspeed.evolution.registry import EvolutionRecord, EvolutionRegistry
from godspeed.evolution.safety import SafetyVerdict

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_candidate(artifact_id: str = "file_read") -> MutationCandidate:
    return MutationCandidate(
        artifact_type="tool_description",
        artifact_id=artifact_id,
        original="Read files.",
        mutated="Read files with path validation.",
        mutation_rationale="improve clarity",
        model_used="ollama/test",
    )


def _make_score(overall: float = 0.8) -> FitnessScore:
    return FitnessScore(
        correctness=0.9,
        procedure_following=0.8,
        conciseness=0.7,
        overall=overall,
        length_penalty=0.0,
        confidence=1.0,
    )


def _make_verdict(passed: bool = True) -> SafetyVerdict:
    return SafetyVerdict(
        passed=passed,
        checks=(("size_limit", True, "ok"), ("fitness_threshold", passed, "ok")),
        requires_human_review=False,
    )


# ---------------------------------------------------------------------------
# Test: register
# ---------------------------------------------------------------------------


class TestRegister:
    def test_register_returns_id(self, tmp_path: Path) -> None:
        registry = EvolutionRegistry(tmp_path)
        record_id = registry.register(_make_candidate(), _make_score(), _make_verdict())
        assert isinstance(record_id, str)
        assert len(record_id) > 0

    def test_creates_registry_file(self, tmp_path: Path) -> None:
        registry = EvolutionRegistry(tmp_path)
        registry.register(_make_candidate(), _make_score(), _make_verdict())
        assert (tmp_path / "registry.jsonl").exists()

    def test_creates_candidate_file(self, tmp_path: Path) -> None:
        registry = EvolutionRegistry(tmp_path)
        record_id = registry.register(_make_candidate(), _make_score(), _make_verdict())
        assert (tmp_path / "candidates" / f"{record_id}.json").exists()

    def test_multiple_registrations(self, tmp_path: Path) -> None:
        registry = EvolutionRegistry(tmp_path)
        id1 = registry.register(_make_candidate("tool_a"), _make_score(), _make_verdict())
        id2 = registry.register(_make_candidate("tool_b"), _make_score(), _make_verdict())
        assert id1 != id2

        records = registry.get_history("tool_a")
        assert len(records) == 1
        assert records[0].artifact_id == "tool_a"


# ---------------------------------------------------------------------------
# Test: mark_applied / mark_reverted
# ---------------------------------------------------------------------------


class TestApplyRevert:
    def test_mark_applied(self, tmp_path: Path) -> None:
        registry = EvolutionRegistry(tmp_path)
        record_id = registry.register(_make_candidate(), _make_score(), _make_verdict())

        registry.mark_applied(record_id, "mutated text")
        record = registry.get_record(record_id)

        assert record is not None
        assert record.applied_at != ""
        assert (tmp_path / "applied" / "file_read.json").exists()

    def test_mark_reverted(self, tmp_path: Path) -> None:
        registry = EvolutionRegistry(tmp_path)
        record_id = registry.register(_make_candidate(), _make_score(), _make_verdict())
        registry.mark_applied(record_id, "mutated text")
        registry.mark_reverted(record_id)

        record = registry.get_record(record_id)
        assert record is not None
        assert record.reverted_at != ""
        assert not (tmp_path / "applied" / "file_read.json").exists()


# ---------------------------------------------------------------------------
# Test: originals backup
# ---------------------------------------------------------------------------


class TestOriginals:
    def test_save_and_retrieve(self, tmp_path: Path) -> None:
        registry = EvolutionRegistry(tmp_path)
        registry.save_original("file_read", "Original description.")
        result = registry.get_original("file_read")
        assert result == "Original description."

    def test_only_saves_first(self, tmp_path: Path) -> None:
        registry = EvolutionRegistry(tmp_path)
        registry.save_original("file_read", "First original.")
        registry.save_original("file_read", "Second original.")  # Should not overwrite
        result = registry.get_original("file_read")
        assert result == "First original."

    def test_missing_original(self, tmp_path: Path) -> None:
        registry = EvolutionRegistry(tmp_path)
        assert registry.get_original("nonexistent") is None


# ---------------------------------------------------------------------------
# Test: get_applied / list_applied
# ---------------------------------------------------------------------------


class TestAppliedQueries:
    def test_get_applied(self, tmp_path: Path) -> None:
        registry = EvolutionRegistry(tmp_path)
        record_id = registry.register(_make_candidate(), _make_score(), _make_verdict())
        registry.mark_applied(record_id, "mutated text")

        applied = registry.get_applied("file_read")
        assert applied is not None
        assert applied["record_id"] == record_id
        assert applied["mutated_text"] == "mutated text"

    def test_get_applied_missing(self, tmp_path: Path) -> None:
        registry = EvolutionRegistry(tmp_path)
        assert registry.get_applied("nonexistent") is None

    def test_list_applied(self, tmp_path: Path) -> None:
        registry = EvolutionRegistry(tmp_path)
        id1 = registry.register(_make_candidate("tool_a"), _make_score(), _make_verdict())
        id2 = registry.register(_make_candidate("tool_b"), _make_score(), _make_verdict())
        registry.mark_applied(id1, "text_a")
        registry.mark_applied(id2, "text_b")

        applied = registry.list_applied()
        assert len(applied) == 2


# ---------------------------------------------------------------------------
# Test: get_candidate
# ---------------------------------------------------------------------------


class TestGetCandidate:
    def test_retrieves_candidate(self, tmp_path: Path) -> None:
        registry = EvolutionRegistry(tmp_path)
        record_id = registry.register(_make_candidate(), _make_score(), _make_verdict())

        candidate = registry.get_candidate(record_id)
        assert candidate is not None
        assert candidate["artifact_id"] == "file_read"
        assert candidate["original"] == "Read files."
        assert candidate["mutated"] == "Read files with path validation."

    def test_missing_candidate(self, tmp_path: Path) -> None:
        registry = EvolutionRegistry(tmp_path)
        assert registry.get_candidate("nonexistent") is None


# ---------------------------------------------------------------------------
# Test: leaderboard
# ---------------------------------------------------------------------------


class TestLeaderboard:
    def test_returns_top_applied(self, tmp_path: Path) -> None:
        registry = EvolutionRegistry(tmp_path)

        # Register and apply 3 mutations with different fitness
        for i, score_val in enumerate([0.6, 0.9, 0.7]):
            record_id = registry.register(
                _make_candidate(f"tool_{i}"),
                _make_score(overall=score_val),
                _make_verdict(),
            )
            registry.mark_applied(record_id, f"text_{i}")

        leaders = registry.leaderboard(top_n=2)
        assert len(leaders) == 2
        assert leaders[0].fitness_overall == 0.9
        assert leaders[1].fitness_overall == 0.7

    def test_excludes_reverted(self, tmp_path: Path) -> None:
        registry = EvolutionRegistry(tmp_path)
        record_id = registry.register(_make_candidate(), _make_score(), _make_verdict())
        registry.mark_applied(record_id, "text")
        registry.mark_reverted(record_id)

        leaders = registry.leaderboard()
        assert len(leaders) == 0


# ---------------------------------------------------------------------------
# Test: stats
# ---------------------------------------------------------------------------


class TestStats:
    def test_empty_registry(self, tmp_path: Path) -> None:
        registry = EvolutionRegistry(tmp_path)
        s = registry.stats()
        assert s["total_mutations"] == 0
        assert s["avg_fitness"] == 0.0

    def test_populated_registry(self, tmp_path: Path) -> None:
        registry = EvolutionRegistry(tmp_path)
        id1 = registry.register(_make_candidate("a"), _make_score(0.8), _make_verdict(True))
        registry.register(_make_candidate("b"), _make_score(0.4), _make_verdict(False))
        registry.mark_applied(id1, "text")

        s = registry.stats()
        assert s["total_mutations"] == 2
        assert s["applied"] == 1
        assert s["safety_passed"] == 1
        assert s["safety_failed"] == 1
        assert s["avg_fitness"] == pytest.approx(0.6, abs=0.01)


# ---------------------------------------------------------------------------
# Test: EvolutionRecord
# ---------------------------------------------------------------------------


class TestEvolutionRecord:
    def test_to_dict_from_dict_roundtrip(self) -> None:
        rec = EvolutionRecord(
            record_id="test-123",
            artifact_type="tool_description",
            artifact_id="bash",
            original_hash="abc123",
            mutated_hash="def456",
            fitness_overall=0.85,
            fitness_confidence=1.0,
            safety_passed=True,
            requires_review=False,
            model_used="ollama/test",
            created_at="2026-04-13T00:00:00+00:00",
            applied_at="",
            reverted_at="",
        )
        d = rec.to_dict()
        rec2 = EvolutionRecord.from_dict(d)
        assert rec.record_id == rec2.record_id
        assert rec.fitness_overall == rec2.fitness_overall
