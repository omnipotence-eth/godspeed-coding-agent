"""Tests for the auto-injected repo-map summary."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from godspeed.context.repo_summary import (
    MAX_SUMMARY_CHARS,
    MIN_FILES_FOR_INJECTION,
    build_repo_summary,
)


def _make_python_file(path: Path, body: str = "def f():\n    return 1\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)


class TestBuildRepoSummary:
    def test_returns_none_for_missing_dir(self, tmp_path: Path) -> None:
        assert build_repo_summary(tmp_path / "does_not_exist") is None

    def test_returns_none_for_small_project(self, tmp_path: Path) -> None:
        # One file is not enough to justify injecting a summary.
        _make_python_file(tmp_path / "only.py")
        assert build_repo_summary(tmp_path) is None

    def test_returns_summary_when_above_threshold(self, tmp_path: Path) -> None:
        # Create just enough source files to trip the threshold.
        for i in range(MIN_FILES_FOR_INJECTION):
            _make_python_file(tmp_path / f"src/module_{i}.py")
        result = build_repo_summary(tmp_path)
        assert result is not None
        assert len(result) > 0

    def test_truncates_oversized_summary(self, tmp_path: Path) -> None:
        # Craft a summary larger than MAX_SUMMARY_CHARS by stubbing
        # RepoMapper — cheaper than building a real huge tree.
        for i in range(MIN_FILES_FOR_INJECTION + 1):
            _make_python_file(tmp_path / f"src/module_{i}.py")

        oversized = "x" * (MAX_SUMMARY_CHARS + 5000)

        class _FakeMapper:
            available = True

            def map_directory(self, directory: Path, max_depth: int = 5, pattern: str = "") -> str:
                return oversized

        with patch("godspeed.context.repo_map.RepoMapper", _FakeMapper):
            result = build_repo_summary(tmp_path)
        assert result is not None
        assert len(result) <= MAX_SUMMARY_CHARS + 100  # truncation note can overshoot slightly
        assert "truncated" in result

    def test_fallback_when_tree_sitter_unavailable(self, tmp_path: Path) -> None:
        # Stub the mapper as unavailable — we should still get a
        # useful file-list fallback.
        for i in range(MIN_FILES_FOR_INJECTION + 1):
            _make_python_file(tmp_path / f"src/mod_{i}.py")
        _make_python_file(tmp_path / "tests/test_a.py")

        class _UnavailableMapper:
            available = False

            def map_directory(self, directory: Path, max_depth: int = 5, pattern: str = "") -> str:
                return "tree-sitter not available"

        with patch("godspeed.context.repo_map.RepoMapper", _UnavailableMapper):
            result = build_repo_summary(tmp_path)
        assert result is not None
        # Fallback listing should include top-level dirs we created.
        assert "src/" in result
        assert "tests/" in result

    def test_skips_excluded_dirs_in_count(self, tmp_path: Path) -> None:
        # node_modules / .venv etc. should NOT inflate the file count
        # past the injection threshold.
        for i in range(MIN_FILES_FOR_INJECTION + 5):
            _make_python_file(tmp_path / f"node_modules/pkg/file_{i}.py")
        # Exactly one real source file in the project — below threshold.
        _make_python_file(tmp_path / "src/real.py")
        assert build_repo_summary(tmp_path) is None

    def test_never_raises_on_filesystem_errors(self, tmp_path: Path) -> None:
        # Permission errors / weird races during summary construction
        # should degrade gracefully to None, not crash session start.
        for i in range(MIN_FILES_FOR_INJECTION + 1):
            _make_python_file(tmp_path / f"src/m_{i}.py")

        class _BrokenMapper:
            available = True

            def map_directory(self, directory: Path, max_depth: int = 5, pattern: str = "") -> str:
                raise RuntimeError("simulated failure")

        with patch("godspeed.context.repo_map.RepoMapper", _BrokenMapper):
            # Should not raise — should fall back to listing.
            result = build_repo_summary(tmp_path)
        # Either fallback-listing succeeded, or we got None. Both are
        # acceptable — the guarantee is "never raises."
        if result is not None:
            assert isinstance(result, str)
