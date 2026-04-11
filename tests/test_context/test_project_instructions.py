"""Tests for context management — project instructions and compaction."""

from __future__ import annotations

from pathlib import Path

from godspeed.context.project_instructions import (
    find_project_root,
    load_project_instructions,
)


class TestLoadProjectInstructions:
    """Test GODSPEED.md loading."""

    def test_load_from_cwd(self, tmp_path: Path) -> None:
        (tmp_path / "GODSPEED.md").write_text("# Project instructions\nUse pytest.")
        result = load_project_instructions(tmp_path)
        assert result is not None
        assert "Use pytest" in result

    def test_no_file_returns_none(self, tmp_path: Path) -> None:
        result = load_project_instructions(tmp_path, walk_parents=False)
        assert result is None

    def test_empty_file_returns_none(self, tmp_path: Path) -> None:
        (tmp_path / "GODSPEED.md").write_text("")
        result = load_project_instructions(tmp_path, walk_parents=False)
        assert result is None

    def test_walk_parents(self, tmp_path: Path) -> None:
        (tmp_path / "GODSPEED.md").write_text("Parent instructions")
        child = tmp_path / "src"
        child.mkdir()
        result = load_project_instructions(child, walk_parents=True)
        assert result is not None
        assert "Parent instructions" in result

    def test_no_walk_parents(self, tmp_path: Path) -> None:
        (tmp_path / "GODSPEED.md").write_text("Parent instructions")
        child = tmp_path / "src"
        child.mkdir()
        result = load_project_instructions(child, walk_parents=False)
        assert result is None

    def test_multiple_files_concatenated(self, tmp_path: Path) -> None:
        (tmp_path / "GODSPEED.md").write_text("Root instructions")
        child = tmp_path / "src"
        child.mkdir()
        (child / "GODSPEED.md").write_text("Src instructions")
        result = load_project_instructions(child, walk_parents=True)
        assert result is not None
        assert "Root instructions" in result
        assert "Src instructions" in result

    def test_custom_filename(self, tmp_path: Path) -> None:
        (tmp_path / "CUSTOM.md").write_text("Custom instructions")
        result = load_project_instructions(tmp_path, filename="CUSTOM.md")
        assert result is not None
        assert "Custom instructions" in result


class TestFindProjectRoot:
    """Test project root detection."""

    def test_git_marker(self, tmp_path: Path) -> None:
        (tmp_path / ".git").mkdir()
        child = tmp_path / "src" / "pkg"
        child.mkdir(parents=True)
        assert find_project_root(child) == tmp_path

    def test_pyproject_marker(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'")
        child = tmp_path / "src"
        child.mkdir()
        assert find_project_root(child) == tmp_path

    def test_godspeed_dir_marker(self, tmp_path: Path) -> None:
        (tmp_path / ".godspeed").mkdir()
        child = tmp_path / "deep" / "nested"
        child.mkdir(parents=True)
        assert find_project_root(child) == tmp_path

    def test_no_marker_returns_cwd(self, tmp_path: Path) -> None:
        child = tmp_path / "orphan"
        child.mkdir()
        # Use a custom marker that won't exist anywhere
        assert find_project_root(child, markers=("__nonexistent_marker__",)) == child

    def test_custom_markers(self, tmp_path: Path) -> None:
        (tmp_path / "Makefile").write_text("all:")
        child = tmp_path / "src"
        child.mkdir()
        assert find_project_root(child, markers=("Makefile",)) == tmp_path
