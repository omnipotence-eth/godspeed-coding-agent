"""Shared directory exclusion patterns for search tools."""

from __future__ import annotations

from pathlib import Path

DEFAULT_EXCLUDES = frozenset(
    {
        "node_modules",
        ".venv",
        "venv",
        "__pycache__",
        ".git",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        ".hypothesis",
        ".tox",
        "dist",
        "build",
        ".eggs",
        "htmlcov",
        ".coverage",
    }
)


def is_excluded(path: Path, excludes: frozenset[str] = DEFAULT_EXCLUDES) -> bool:
    """Check if any path component matches an exclude pattern."""
    return any(part in excludes for part in path.parts)
