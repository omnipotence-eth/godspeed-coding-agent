"""Tests for skill discovery and loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from godspeed.skills.loader import SkillDefinition, _parse_skill_file, discover_skills


@pytest.fixture()
def skill_dir(tmp_path: Path) -> Path:
    """Create a temp directory with valid skill files."""
    skills = tmp_path / "skills"
    skills.mkdir()
    return skills


def _write_skill(directory: Path, filename: str, content: str) -> Path:
    """Write a skill file and return its path."""
    path = directory / filename
    path.write_text(content, encoding="utf-8")
    return path


class TestParseSkillFile:
    """Test _parse_skill_file()."""

    def test_valid_skill(self, skill_dir: Path) -> None:
        path = _write_skill(
            skill_dir,
            "review.md",
            "---\nname: code-review\ndescription: Review code\n"
            "trigger: review\n---\nReview this code.",
        )
        skill = _parse_skill_file(path)
        assert skill is not None
        assert skill.name == "code-review"
        assert skill.description == "Review code"
        assert skill.trigger == "review"
        assert skill.content == "Review this code."

    def test_multiline_content(self, skill_dir: Path) -> None:
        path = _write_skill(
            skill_dir,
            "multi.md",
            "---\nname: multi\ndescription: Multi-line\n"
            "trigger: multi\n---\nLine 1\n\nLine 2\nLine 3",
        )
        skill = _parse_skill_file(path)
        assert skill is not None
        assert "Line 1" in skill.content
        assert "Line 3" in skill.content

    def test_missing_frontmatter(self, skill_dir: Path) -> None:
        path = _write_skill(skill_dir, "bad.md", "No frontmatter here.")
        assert _parse_skill_file(path) is None

    def test_missing_closing_marker(self, skill_dir: Path) -> None:
        path = _write_skill(skill_dir, "bad.md", "---\nname: test\n\nNo closing marker")
        assert _parse_skill_file(path) is None

    def test_missing_required_fields(self, skill_dir: Path) -> None:
        path = _write_skill(
            skill_dir,
            "bad.md",
            "---\nname: test\n---\nContent without description or trigger.",
        )
        assert _parse_skill_file(path) is None

    def test_empty_body(self, skill_dir: Path) -> None:
        path = _write_skill(
            skill_dir,
            "empty.md",
            "---\nname: empty\ndescription: Empty body\ntrigger: empty\n---\n",
        )
        assert _parse_skill_file(path) is None

    def test_invalid_yaml(self, skill_dir: Path) -> None:
        path = _write_skill(
            skill_dir,
            "bad_yaml.md",
            "---\n[invalid yaml: {{{\n---\nContent.",
        )
        assert _parse_skill_file(path) is None

    def test_non_dict_frontmatter(self, skill_dir: Path) -> None:
        path = _write_skill(
            skill_dir,
            "list.md",
            "---\n- item1\n- item2\n---\nContent.",
        )
        assert _parse_skill_file(path) is None

    def test_nonexistent_file(self, tmp_path: Path) -> None:
        assert _parse_skill_file(tmp_path / "nonexistent.md") is None

    def test_fields_coerced_to_string(self, skill_dir: Path) -> None:
        path = _write_skill(
            skill_dir,
            "numeric.md",
            "---\nname: 123\ndescription: 456\ntrigger: 789\n---\nContent.",
        )
        skill = _parse_skill_file(path)
        assert skill is not None
        assert skill.name == "123"
        assert skill.trigger == "789"


class TestDiscoverSkills:
    """Test discover_skills()."""

    def test_empty_directory(self, skill_dir: Path) -> None:
        skills = discover_skills([skill_dir])
        assert skills == []

    def test_discovers_valid_skills(self, skill_dir: Path) -> None:
        _write_skill(
            skill_dir,
            "review.md",
            "---\nname: review\ndescription: Review code\ntrigger: review\n---\nReview code.",
        )
        _write_skill(
            skill_dir,
            "test.md",
            "---\nname: test\ndescription: Run tests\ntrigger: test\n---\nRun tests.",
        )
        skills = discover_skills([skill_dir])
        assert len(skills) == 2
        triggers = {s.trigger for s in skills}
        assert triggers == {"review", "test"}

    def test_skips_invalid_files(self, skill_dir: Path) -> None:
        _write_skill(
            skill_dir,
            "good.md",
            "---\nname: good\ndescription: Good\ntrigger: good\n---\nContent.",
        )
        _write_skill(skill_dir, "bad.md", "No frontmatter")
        skills = discover_skills([skill_dir])
        assert len(skills) == 1
        assert skills[0].trigger == "good"

    def test_project_overrides_global(self, tmp_path: Path) -> None:
        global_dir = tmp_path / "global"
        project_dir = tmp_path / "project"
        global_dir.mkdir()
        project_dir.mkdir()

        _write_skill(
            global_dir,
            "review.md",
            "---\nname: global-review\ndescription: Global\ntrigger: review\n---\nGlobal review.",
        )
        _write_skill(
            project_dir,
            "review.md",
            "---\nname: project-review\ndescription: Project\n"
            "trigger: review\n---\nProject review.",
        )

        skills = discover_skills([global_dir, project_dir])
        assert len(skills) == 1
        assert skills[0].name == "project-review"
        assert skills[0].content == "Project review."

    def test_nonexistent_directory(self, tmp_path: Path) -> None:
        skills = discover_skills([tmp_path / "nonexistent"])
        assert skills == []

    def test_ignores_non_md_files(self, skill_dir: Path) -> None:
        _write_skill(
            skill_dir,
            "good.md",
            "---\nname: good\ndescription: Good\ntrigger: good\n---\nContent.",
        )
        (skill_dir / "readme.txt").write_text("Not a skill", encoding="utf-8")
        (skill_dir / "script.py").write_text("print('hello')", encoding="utf-8")
        skills = discover_skills([skill_dir])
        assert len(skills) == 1

    def test_multiple_dirs_merge(self, tmp_path: Path) -> None:
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()

        _write_skill(
            dir_a,
            "review.md",
            "---\nname: review\ndescription: Review\ntrigger: review\n---\nReview.",
        )
        _write_skill(
            dir_b,
            "deploy.md",
            "---\nname: deploy\ndescription: Deploy\ntrigger: deploy\n---\nDeploy.",
        )

        skills = discover_skills([dir_a, dir_b])
        assert len(skills) == 2
        triggers = {s.trigger for s in skills}
        assert triggers == {"review", "deploy"}


class TestSkillDefinition:
    """Test SkillDefinition dataclass."""

    def test_frozen(self) -> None:
        skill = SkillDefinition(name="test", description="desc", trigger="test", content="body")
        with pytest.raises(AttributeError):
            skill.name = "changed"  # type: ignore[misc]

    def test_equality(self) -> None:
        a = SkillDefinition(name="test", description="desc", trigger="t", content="c")
        b = SkillDefinition(name="test", description="desc", trigger="t", content="c")
        assert a == b
