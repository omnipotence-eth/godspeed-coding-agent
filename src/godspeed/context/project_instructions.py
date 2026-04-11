"""GODSPEED.md project instructions loader.

Walks up the directory tree loading applicable instruction files,
similar to Claude Code's CLAUDE.md pattern.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_FILENAME = "GODSPEED.md"


def load_project_instructions(
    cwd: Path,
    filename: str = DEFAULT_FILENAME,
    walk_parents: bool = True,
) -> str | None:
    """Load project instructions from GODSPEED.md.

    Searches from cwd upward through parent directories.
    If multiple files found, they are concatenated (parent first, child last)
    so the most specific instructions take precedence.

    Args:
        cwd: Starting directory to search from.
        filename: Instruction file name (default: GODSPEED.md).
        walk_parents: Whether to walk up the directory tree.

    Returns:
        Combined instructions text, or None if no files found.
    """
    found_files: list[tuple[Path, str]] = []

    # Walk up directory tree
    current = cwd.resolve()
    root = current.anchor

    while True:
        instructions_path = current / filename
        if instructions_path.is_file():
            try:
                content = instructions_path.read_text(encoding="utf-8").strip()
                if content:
                    found_files.append((instructions_path, content))
                    logger.info("Found project instructions at %s", instructions_path)
            except OSError as exc:
                logger.warning(
                    "Failed to read instructions %s: %s",
                    instructions_path,
                    exc,
                )

        if not walk_parents or str(current) == root:
            break
        parent = current.parent
        if parent == current:
            break
        current = parent

    if not found_files:
        return None

    # Reverse so parent instructions come first (most general → most specific)
    found_files.reverse()

    if len(found_files) == 1:
        return found_files[0][1]

    # Concatenate with source headers
    parts = []
    for path, content in found_files:
        parts.append(f"# From: {path}\n\n{content}")
    return "\n\n---\n\n".join(parts)


def find_project_root(cwd: Path, markers: tuple[str, ...] | None = None) -> Path:
    """Find the project root by looking for common markers.

    Args:
        cwd: Starting directory.
        markers: Files/dirs that indicate project root.

    Returns:
        Project root path, or cwd if no markers found.
    """
    if markers is None:
        markers = (
            ".git",
            "pyproject.toml",
            "package.json",
            "Cargo.toml",
            "go.mod",
            ".godspeed",
            "GODSPEED.md",
        )

    current = cwd.resolve()
    while True:
        for marker in markers:
            if (current / marker).exists():
                return current
        parent = current.parent
        if parent == current:
            break
        current = parent

    return cwd
