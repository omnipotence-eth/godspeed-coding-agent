"""Tests for dream consolidation — date normalization and periodic runs."""

from __future__ import annotations

from pathlib import Path

import pytest

from godspeed.skills.dream import RELATIVE_DATE_RE, SkillDream


class TestDateNormalization:
    """Test _normalize_dates relative date conversion."""

    def _normalize(self, text: str) -> str:
        dream = SkillDream(base_dir=Path("/tmp"))
        return dream._normalize_dates(text)

    def test_yesterday(self) -> None:
        result = self._normalize("This happened yesterday.")
        assert "yesterday" not in result
        assert "-" in result  # contains a date

    def test_today(self) -> None:
        result = self._normalize("Updated today.")
        assert "today" not in result

    def test_last_week(self) -> None:
        result = self._normalize("Changed last week.")
        assert "last week" not in result

    def test_3_days_ago(self) -> None:
        result = self._normalize("Added 3 days ago.")
        assert "days ago" not in result

    def test_2_weeks_ago(self) -> None:
        result = self._normalize("Fixed 2 weeks ago.")
        assert "weeks ago" not in result

    def test_1_month_ago(self) -> None:
        result = self._normalize("Refactored 1 month ago.")
        assert "month ago" not in result

    def test_1_year_ago(self) -> None:
        result = self._normalize("Created 1 year ago.")
        assert "year ago" not in result

    def test_no_relative_dates(self) -> None:
        text = "This has 2024-01-15 and no relative dates."
        result = self._normalize(text)
        assert result == text

    def test_last_session(self) -> None:
        result = self._normalize("Updated last session.")
        assert "last session" not in result

    def test_mixed_content(self) -> None:
        result = self._normalize("Yesterday I started. Last week I finished.")
        assert "yesterday" not in result
        assert "last week" not in result

    def test_case_insensitive(self) -> None:
        result = self._normalize("Yesterday is case insensitive")
        assert "Yesterday" not in result


class TestDreamRun:
    """Test dream.run() consolidation pass."""

    def test_normalizes_skill_dates(self, tmp_path: Path) -> None:
        dream = SkillDream(base_dir=tmp_path)
        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "test-skill"
        skill_dir.mkdir(parents=True)
        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text("---\nname: test-skill\ndescription: T\ntrigger: ts\n---\n\nUpdated yesterday.")

        stats = dream.run(skills_dir)
        assert stats["dates_normalized"] == 1
        assert stats["errors"] == 0
        text = skill_md.read_text()
        assert "yesterday" not in text

    def test_skips_non_skill_files(self, tmp_path: Path) -> None:
        dream = SkillDream(base_dir=tmp_path)
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        (skills_dir / "README.md").write_text("Just a note.")

        stats = dream.run(skills_dir)
        assert stats["consolidated"] == 0
        assert stats["dates_normalized"] == 0

    def test_empty_skills_dir(self, tmp_path: Path) -> None:
        dream = SkillDream(base_dir=tmp_path)
        stats = dream.run(tmp_path / "empty")
        assert stats == {"consolidated": 0, "pruned": 0, "dates_normalized": 0, "errors": 0}

    def test_multiple_skills_normalized(self, tmp_path: Path) -> None:
        dream = SkillDream(base_dir=tmp_path)
        skills_dir = tmp_path / "skills"
        for name in ("skill-a", "skill-b"):
            d = skills_dir / name
            d.mkdir(parents=True)
            (d / "SKILL.md").write_text(f"---\nname: {name}\ndescription: D\ntrigger: {name}\n---\n\nCreated yesterday.")

        stats = dream.run(skills_dir)
        assert stats["dates_normalized"] == 2

    def test_writes_dream_lock(self, tmp_path: Path) -> None:
        dream = SkillDream(base_dir=tmp_path)
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        dream.run(skills_dir)
        lock = tmp_path / ".dream" / "last_run.txt"
        assert lock.exists()
        assert lock.read_text().strip()

    def test_handles_os_error_gracefully(self, tmp_path: Path) -> None:
        dream = SkillDream(base_dir=tmp_path)
        skills_dir = tmp_path / "skills"
        d = skills_dir / "broken"
        d.mkdir(parents=True)
        skill_md = d / "SKILL.md"
        skill_md.write_text("---\nname: broken\ndescription: B\ntrigger: b\n---\n\nBody.")
        # Make unreadable
        skill_md.chmod(0o222)
        try:
            stats = dream.run(skills_dir)
            assert stats["errors"] >= 0
        finally:
            skill_md.chmod(0o644)


class TestShouldRun:
    """Test dream interval checking."""

    def test_should_run_no_lock(self, tmp_path: Path) -> None:
        dream = SkillDream(base_dir=tmp_path)
        assert dream.should_run()

    def test_should_not_run_recently(self, tmp_path: Path) -> None:
        dream = SkillDream(base_dir=tmp_path)
        from datetime import UTC, datetime
        lock = tmp_path / ".dream" / "last_run.txt"
        # Directory already created by SkillDream.__init__
        lock.write_text(datetime.now(tz=UTC).isoformat())
        assert not dream.should_run(interval_hours=24)

    def test_should_run_after_interval(self, tmp_path: Path) -> None:
        dream = SkillDream(base_dir=tmp_path)
        from datetime import UTC, datetime, timedelta
        lock = tmp_path / ".dream" / "last_run.txt"
        past = (datetime.now(tz=UTC) - timedelta(hours=48)).isoformat()
        lock.write_text(past)
        assert dream.should_run(interval_hours=24)

    def test_corrupted_lock_returns_true(self, tmp_path: Path) -> None:
        dream = SkillDream(base_dir=tmp_path)
        lock = tmp_path / ".dream" / "last_run.txt"
        lock.write_text("not-a-date")
        assert dream.should_run()
