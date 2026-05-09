"""Tests for skill evolution — lesson tracking, merging, and skill rewriting."""

from __future__ import annotations

from pathlib import Path

import pytest

from godspeed.skills.evolution import Lesson, SkillEvolution


class TestLessonTracking:
    """Test recording and retrieving lessons."""

    def test_record_and_load(self, tmp_path: Path) -> None:
        evo = SkillEvolution(base_dir=tmp_path / "evo")
        evo.record_lesson(Lesson(skill_name="test", text="Use ruff for linting", confidence=0.8))
        lessons = evo._load_lessons("test")
        assert len(lessons) == 1
        assert lessons[0]["text"] == "Use ruff for linting"

    def test_multiple_lessons_accumulate(self, tmp_path: Path) -> None:
        evo = SkillEvolution(base_dir=tmp_path / "evo")
        evo.record_lesson(Lesson(skill_name="test", text="Lesson 1"))
        evo.record_lesson(Lesson(skill_name="test", text="Lesson 2"))
        lessons = evo._load_lessons("test")
        assert len(lessons) == 2

    def test_lessons_separate_per_skill(self, tmp_path: Path) -> None:
        evo = SkillEvolution(base_dir=tmp_path / "evo")
        evo.record_lesson(Lesson(skill_name="a", text="A lesson"))
        evo.record_lesson(Lesson(skill_name="b", text="B lesson"))
        assert len(evo._load_lessons("a")) == 1
        assert len(evo._load_lessons("b")) == 1

    def test_empty_skill_returns_empty_list(self, tmp_path: Path) -> None:
        evo = SkillEvolution(base_dir=tmp_path / "evo")
        assert evo._load_lessons("nonexistent") == []


class TestMergeDuplicates:
    """Test _merge_duplicates deduplication."""

    def test_merges_identical_lessons(self, tmp_path: Path) -> None:
        evo = SkillEvolution(base_dir=tmp_path / "evo")
        evo.record_lesson(Lesson(skill_name="test", text="Do X", confidence=1.0))
        evo.record_lesson(Lesson(skill_name="test", text="Do X", confidence=0.5))
        merged = evo._merge_duplicates(evo._load_lessons("test"))
        assert len(merged) == 1
        assert merged[0]["confidence"] == 1.5

    def test_case_insensitive_merge(self, tmp_path: Path) -> None:
        evo = SkillEvolution(base_dir=tmp_path / "evo")
        evo.record_lesson(Lesson(skill_name="test", text="Do X", confidence=1.0))
        evo.record_lesson(Lesson(skill_name="test", text="do x", confidence=0.5))
        merged = evo._merge_duplicates(evo._load_lessons("test"))
        assert len(merged) == 1

    def test_different_texts_kept_separate(self, tmp_path: Path) -> None:
        evo = SkillEvolution(base_dir=tmp_path / "evo")
        evo.record_lesson(Lesson(skill_name="test", text="Lesson A", confidence=1.0))
        evo.record_lesson(Lesson(skill_name="test", text="Lesson B", confidence=1.0))
        merged = evo._merge_duplicates(evo._load_lessons("test"))
        assert len(merged) == 2


class TestEligibleLessons:
    """Test get_eligible_lessons confidence threshold."""

    def test_above_threshold(self, tmp_path: Path) -> None:
        evo = SkillEvolution(base_dir=tmp_path / "evo")
        evo.record_lesson(Lesson(skill_name="test", text="Important", confidence=1.0))
        evo.record_lesson(Lesson(skill_name="test", text="Important", confidence=0.8))
        eligible = evo.get_eligible_lessons("test", min_confidence=1.5)
        assert len(eligible) == 1
        assert eligible[0]["text"] == "Important"

    def test_below_threshold(self, tmp_path: Path) -> None:
        evo = SkillEvolution(base_dir=tmp_path / "evo")
        evo.record_lesson(Lesson(skill_name="test", text="Trivial", confidence=0.3))
        eligible = evo.get_eligible_lessons("test", min_confidence=1.5)
        assert eligible == []

    def test_no_lessons(self, tmp_path: Path) -> None:
        evo = SkillEvolution(base_dir=tmp_path / "evo")
        assert evo.get_eligible_lessons("nonexistent") == []

    def test_empty_confidence(self, tmp_path: Path) -> None:
        evo = SkillEvolution(base_dir=tmp_path / "evo")
        evo.record_lesson(Lesson(skill_name="test", text="No confidence"))
        eligible = evo.get_eligible_lessons("test", min_confidence=1.5)
        assert eligible == []


class TestEvolve:
    """Test evolve() rewriting of SKILL.md."""

    def _setup_skill(self, tmp_path: Path, name: str = "test-skill") -> Path:
        skill_dir = tmp_path / "skills" / name
        skill_dir.mkdir(parents=True)
        skill_path = skill_dir / "SKILL.md"
        skill_path.write_text("---\nname: test-skill\ndescription: T\ntrigger: ts\n---\n\nOriginal content.")
        return skill_path

    def test_appends_lessons_section(self, tmp_path: Path) -> None:
        evo = SkillEvolution(base_dir=tmp_path / "evo")
        skill_path = self._setup_skill(tmp_path)
        evo.record_lesson(Lesson(skill_name="test-skill", text="Use async patterns", confidence=2.0))
        updated = evo.evolve("test-skill", skill_path)
        assert updated
        text = skill_path.read_text()
        assert "## Lessons" in text
        assert "Use async patterns" in text

    def test_creates_backup(self, tmp_path: Path) -> None:
        evo = SkillEvolution(base_dir=tmp_path / "evo")
        skill_path = self._setup_skill(tmp_path)
        evo.record_lesson(Lesson(skill_name="test-skill", text="Important lesson", confidence=2.0))
        evo.evolve("test-skill", skill_path)
        backups = list((tmp_path / "evo" / "backups").iterdir())
        assert len(backups) >= 1

    def test_no_eligible_lessons(self, tmp_path: Path) -> None:
        evo = SkillEvolution(base_dir=tmp_path / "evo")
        skill_path = self._setup_skill(tmp_path)
        evo.record_lesson(Lesson(skill_name="test-skill", text="Trivial", confidence=0.1))
        updated = evo.evolve("test-skill", skill_path)
        assert not updated

    def test_does_not_duplicate_lessons_section(self, tmp_path: Path) -> None:
        evo = SkillEvolution(base_dir=tmp_path / "evo")
        skill_path = self._setup_skill(tmp_path)
        evo.record_lesson(Lesson(skill_name="test-skill", text="Lesson A", confidence=2.0))
        evo.evolve("test-skill", skill_path)
        evo.record_lesson(Lesson(skill_name="test-skill", text="Lesson B", confidence=2.0))
        evo.evolve("test-skill", skill_path)
        text = skill_path.read_text()
        assert text.count("## Lessons") == 1

    def test_folded_lessons_removed_from_store(self, tmp_path: Path) -> None:
        evo = SkillEvolution(base_dir=tmp_path / "evo")
        skill_path = self._setup_skill(tmp_path)
        evo.record_lesson(Lesson(skill_name="test-skill", text="Folded lesson", confidence=2.0))
        evo.record_lesson(Lesson(skill_name="test-skill", text="Remaining lesson", confidence=0.5))
        evo.evolve("test-skill", skill_path)
        remaining = evo._load_lessons("test-skill")
        texts = [l["text"] for l in remaining]
        assert "Folded lesson" not in texts
        assert "Remaining lesson" in texts

    def test_missing_skill_file(self, tmp_path: Path) -> None:
        evo = SkillEvolution(base_dir=tmp_path / "evo")
        evo.record_lesson(Lesson(skill_name="ghost", text="Lost", confidence=2.0))
        updated = evo.evolve("ghost", tmp_path / "nonexistent.md")
        assert not updated
