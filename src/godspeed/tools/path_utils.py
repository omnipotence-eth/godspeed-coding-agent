"""Shared path utilities for file tools."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def resolve_tool_path(file_path: str, cwd: Path) -> Path:
    """Resolve a file path relative to the project root.

    Raises:
        ValueError: If the resolved path is outside the project directory.
    """
    path = Path(file_path)
    resolved = path.resolve() if path.is_absolute() else (cwd / path).resolve()
    cwd_resolved = cwd.resolve()

    try:
        resolved.relative_to(cwd_resolved)
    except ValueError as exc:
        raise ValueError(
            f"Access denied: path '{file_path}' resolves to '{resolved}' "
            f"which is outside the project directory '{cwd_resolved}'"
        ) from exc
    return resolved
