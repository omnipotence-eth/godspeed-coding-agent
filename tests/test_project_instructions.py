"""Tests for multi-file project instructions loader."""

from __future__ import annotations

from pathlib import Path

from godspeed.context.project_instructions import (
    INSTRUCTION_FILES,
    _load_single_file,
    load_project_instructions,
)


class TestLoadProjectInstructions:
    """Test the multi-file instruction loader."""

    def test_loads_godspeed_md(self, tmp_path: Path) -> None:
        (tmp_path / "GODSPEED.md").write_text("# Project rules", encoding="utf-8")
        result = load_project_instructions(tmp_path)
        assert result is not None
        assert "Project rules" in result

    def test_loads_agents_md(self, tmp_path: Path) -> None:
        (tmp_path / "AGENTS.md").write_text("# Agents config", encoding="utf-8")
        result = load_project_instructions(tmp_path)
        assert result is not None
        assert "Agents config" in result

    def test_loads_claude_md(self, tmp_path: Path) -> None:
        (tmp_path / "CLAUDE.md").write_text("# Claude config", encoding="utf-8")
        result = load_project_instructions(tmp_path)
        assert result is not None
        assert "Claude config" in result

    def test_loads_cursorrules(self, tmp_path: Path) -> None:
        (tmp_path / ".cursorrules").write_text("cursor rules here", encoding="utf-8")
        result = load_project_instructions(tmp_path)
        assert result is not None
        assert "cursor rules here" in result

    def test_merges_multiple_files(self, tmp_path: Path) -> None:
        (tmp_path / "GODSPEED.md").write_text("Godspeed rules", encoding="utf-8")
        (tmp_path / "AGENTS.md").write_text("Agent rules", encoding="utf-8")
        result = load_project_instructions(tmp_path)
        assert result is not None
        assert "Godspeed rules" in result
        assert "Agent rules" in result
        # Should have separator
        assert "---" in result

    def test_godspeed_md_takes_priority_first(self, tmp_path: Path) -> None:
        """GODSPEED.md should appear before AGENTS.md in merged output."""
        (tmp_path / "GODSPEED.md").write_text("FIRST", encoding="utf-8")
        (tmp_path / "AGENTS.md").write_text("SECOND", encoding="utf-8")
        result = load_project_instructions(tmp_path)
        assert result is not None
        assert result.index("FIRST") < result.index("SECOND")

    def test_no_files_returns_none(self, tmp_path: Path) -> None:
        result = load_project_instructions(tmp_path, walk_parents=False)
        assert result is None

    def test_empty_files_skipped(self, tmp_path: Path) -> None:
        (tmp_path / "GODSPEED.md").write_text("", encoding="utf-8")
        (tmp_path / "AGENTS.md").write_text("Real content", encoding="utf-8")
        result = load_project_instructions(tmp_path)
        assert result is not None
        assert "Real content" in result

    def test_custom_filename_single_mode(self, tmp_path: Path) -> None:
        """Non-default filename triggers single-file mode."""
        (tmp_path / "CUSTOM.md").write_text("Custom", encoding="utf-8")
        (tmp_path / "AGENTS.md").write_text("Should not load", encoding="utf-8")
        result = load_project_instructions(tmp_path, filename="CUSTOM.md")
        assert result is not None
        assert "Custom" in result
        assert "Should not load" not in result

    def test_no_duplicate_paths(self, tmp_path: Path) -> None:
        """Same file should not be loaded twice."""
        (tmp_path / "GODSPEED.md").write_text("Only once", encoding="utf-8")
        result = load_project_instructions(tmp_path)
        assert result is not None
        assert result.count("Only once") == 1

    def test_parent_directory_walk(self, tmp_path: Path) -> None:
        """Instructions from parent dirs should be loaded."""
        (tmp_path / "GODSPEED.md").write_text("Parent rules", encoding="utf-8")
        child = tmp_path / "subdir"
        child.mkdir()
        (child / "GODSPEED.md").write_text("Child rules", encoding="utf-8")
        result = load_project_instructions(child)
        assert result is not None
        assert "Parent rules" in result
        assert "Child rules" in result


class TestInstructionFiles:
    """Test the instruction files constant."""

    def test_godspeed_md_is_first(self) -> None:
        assert INSTRUCTION_FILES[0] == "GODSPEED.md"

    def test_agents_md_included(self) -> None:
        assert "AGENTS.md" in INSTRUCTION_FILES

    def test_claude_md_included(self) -> None:
        assert "CLAUDE.md" in INSTRUCTION_FILES

    def test_cursorrules_included(self) -> None:
        assert ".cursorrules" in INSTRUCTION_FILES


class TestLoadSingleFile:
    """Test the single-file loader."""

    def test_loads_existing_file(self, tmp_path: Path) -> None:
        (tmp_path / "test.md").write_text("content", encoding="utf-8")
        found = _load_single_file(tmp_path, "test.md", walk_parents=False)
        assert len(found) == 1
        assert found[0][1] == "content"

    def test_returns_empty_for_missing(self, tmp_path: Path) -> None:
        found = _load_single_file(tmp_path, "missing.md", walk_parents=False)
        assert len(found) == 0
