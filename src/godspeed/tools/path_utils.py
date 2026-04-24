"""Shared path utilities for file tools."""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)


@lru_cache(maxsize=512)
def resolve_tool_path(file_path: str, cwd: Path) -> Path:
    """Resolve a file path relative to the project root.

    Raises:
        ValueError: If the resolved path is outside the project directory.
    """
    # Reject Windows drive letter paths on any platform (e.g., C:\... or D:/)
    if re.match(r"^[A-Za-z]:[\\\/]", str(file_path)):
        raise ValueError(
            f"Access denied: path '{file_path}' is a Windows absolute path "
            f"which is outside the project directory '{cwd.resolve()}'"
        )

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
