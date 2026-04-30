"""Shared path utilities for file tools."""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def resolve_tool_path(file_path: str, cwd: Path) -> Path:
    """Resolve a file path relative to the project root.

    Includes symlink resolution to prevent symlink traversal attacks.
    Raises:
        ValueError: If the resolved path is outside the project directory.
    """
    path = Path(file_path).expanduser()
    resolved = path.resolve() if path.is_absolute() else (cwd / path).resolve()
    cwd_resolved = cwd.resolve()

    try:
        resolved.relative_to(cwd_resolved)
    except ValueError as exc:
        raise ValueError(
            f"Access denied: path '{file_path}' resolves to '{resolved}' "
            f"which is outside the project directory '{cwd_resolved}'"
        ) from exc

    # Additional symlink protection: resolve symlinks and verify the real path
    # is still within the project directory
    try:
        real_path = Path(os.path.realpath(str(resolved)))
        real_path.relative_to(cwd_resolved)
    except ValueError as exc:
        raise ValueError(
            f"Access denied: path '{file_path}' (real path: '{real_path}') "
            f"resolves via symlinks to outside the project directory '{cwd_resolved}'"
        ) from exc
    except OSError as exc:
        logger.warning("Could not resolve real path for %s: %s", resolved, exc)

    return resolved
