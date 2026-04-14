"""Tests for the evolution applier — runtime hot-swap of evolved artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from godspeed.evolution.applier import EvolutionApplier
from godspeed.evolution.fitness import FitnessScore
from godspeed.evolution.mutator import MutationCandidate
from godspeed.evolution.registry import EvolutionRegistry
from godspeed.evolution.safety import SafetyVerdict

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_candidate(
    artifact_id: str = "file_read",
    artifact_type: str = "tool_description",
) -> MutationCandidate:
    return MutationCandidate(
        artifact_type=artifact_type,
        artifact_id=artifact_id,
        original="Original description.",
        mutated="Improved description.",
        mutation_rationale="test",
        model_used="ollama/test",
    )


def _make_score() -> FitnessScore:
    return FitnessScore(
        correctness=0.9,
        procedure_following=0.8,
        conciseness=0.7,
        overall=0.8,
        length_penalty=0.0,
        confidence=1.0,
    )


def _make_verdict() -> SafetyVerdict:
    return SafetyVerdict(passed=True, checks=(), requires_human_review=False)


def _setup(tmp_path: Path) -> tuple[EvolutionRegistry, EvolutionApplier]:
    registry = EvolutionRegistry(tmp_path)
    applier = EvolutionApplier(registry)
    return registry, applier


# ---------------------------------------------------------------------------
# Test: apply_tool_description
# ---------------------------------------------------------------------------


class TestApplyToolDescription:
    def test_apply(self, tmp_path: Path) -> None:
        registry, applier = _setup(tmp_path)
        record_id = registry.register(_make_candidate(), _make_score(), _make_verdict())

        applier.apply_tool_description(record_id, "file_read", "Improved.", "Original.")
        assert applier.description_overrides["file_read"] == "Improved."
        assert registry.get_applied("file_read") is not None

    def test_get_tool_description_with_override(self, tmp_path: Path) -> None:
        registry, applier = _setup(tmp_path)
        record_id = registry.register(_make_candidate(), _make_score(), _make_verdict())
        applier.apply_tool_description(record_id, "file_read", "Improved.", "Original.")

        assert applier.get_tool_description("file_read", "default") == "Improved."
        assert applier.get_tool_description("bash", "default") == "default"


# ---------------------------------------------------------------------------
# Test: apply_prompt_section
# ---------------------------------------------------------------------------


class TestApplyPromptSection:
    def test_apply(self, tmp_path: Path) -> None:
        registry, applier = _setup(tmp_path)
        candidate = _make_candidate(artifact_id="core", artifact_type="prompt_section")
        record_id = registry.register(candidate, _make_score(), _make_verdict())

        applier.apply_prompt_section(record_id, "core", "Better prompt.", "Old prompt.")
        assert applier.prompt_overrides["core"] == "Better prompt."

    def test_get_prompt_section_with_override(self, tmp_path: Path) -> None:
        registry, applier = _setup(tmp_path)
        candidate = _make_candidate(artifact_id="core", artifact_type="prompt_section")
        record_id = registry.register(candidate, _make_score(), _make_verdict())
        applier.apply_prompt_section(record_id, "core", "Better.", "Old.")

        assert applier.get_prompt_section("core", "default") == "Better."
        assert applier.get_prompt_section("tools", "default") == "default"


# ---------------------------------------------------------------------------
# Test: apply_skill
# ---------------------------------------------------------------------------


class TestApplySkill:
    def test_writes_skill_file(self, tmp_path: Path) -> None:
        registry, applier = _setup(tmp_path)
        candidate = _make_candidate(artifact_id="read-edit", artifact_type="skill")
        record_id = registry.register(candidate, _make_score(), _make_verdict())

        skills_dir = tmp_path / "skills"
        applier.apply_skill(
            record_id,
            "read-edit",
            "---\nname: read-edit\n---\nContent",
            skills_dir,
        )

        skill_path = skills_dir / "read-edit.md"
        assert skill_path.exists()
        assert "read-edit" in skill_path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Test: rollback
# ---------------------------------------------------------------------------


class TestRollback:
    def test_rollback_tool_description(self, tmp_path: Path) -> None:
        registry, applier = _setup(tmp_path)
        record_id = registry.register(_make_candidate(), _make_score(), _make_verdict())
        applier.apply_tool_description(record_id, "file_read", "New.", "Original.")

        assert "file_read" in applier.description_overrides
        result = applier.rollback(record_id)
        assert result is True
        assert "file_read" not in applier.description_overrides

    def test_rollback_prompt_section(self, tmp_path: Path) -> None:
        registry, applier = _setup(tmp_path)
        candidate = _make_candidate(artifact_id="core", artifact_type="prompt_section")
        record_id = registry.register(candidate, _make_score(), _make_verdict())
        applier.apply_prompt_section(record_id, "core", "New.", "Old.")

        result = applier.rollback(record_id)
        assert result is True
        assert "core" not in applier.prompt_overrides

    def test_rollback_nonexistent(self, tmp_path: Path) -> None:
        _, applier = _setup(tmp_path)
        assert applier.rollback("nonexistent") is False

    def test_rollback_unapplied(self, tmp_path: Path) -> None:
        registry, applier = _setup(tmp_path)
        record_id = registry.register(_make_candidate(), _make_score(), _make_verdict())
        # Never applied
        assert applier.rollback(record_id) is False


# ---------------------------------------------------------------------------
# Test: load_applied
# ---------------------------------------------------------------------------


class TestLoadApplied:
    def test_loads_on_startup(self, tmp_path: Path) -> None:
        # First session: apply a mutation
        registry = EvolutionRegistry(tmp_path)
        record_id = registry.register(_make_candidate(), _make_score(), _make_verdict())
        registry.mark_applied(record_id, "Evolved description.")

        # Also manually write the artifact_type to applied json
        applied_path = tmp_path / "applied" / "file_read.json"
        data = json.loads(applied_path.read_text(encoding="utf-8"))
        data["artifact_type"] = "tool_description"
        applied_path.write_text(json.dumps(data), encoding="utf-8")

        # Second session: load from disk
        registry2 = EvolutionRegistry(tmp_path)
        applier2 = EvolutionApplier(registry2)
        count = applier2.load_applied()

        assert count == 1
        assert applier2.description_overrides.get("file_read") == "Evolved description."

    def test_empty_dir(self, tmp_path: Path) -> None:
        registry = EvolutionRegistry(tmp_path)
        applier = EvolutionApplier(registry)
        assert applier.load_applied() == 0
