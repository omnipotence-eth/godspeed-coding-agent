"""Checkpoint save/restore — snapshot conversation state for recovery."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _checkpoints_dir(project_dir: Path) -> Path:
    """Get the checkpoints directory, creating it if needed."""
    d = project_dir / ".godspeed" / "checkpoints"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_checkpoint(
    name: str,
    system_prompt: str,
    messages: list[dict[str, Any]],
    model: str,
    token_count: int,
    project_dir: Path,
) -> Path:
    """Save a conversation checkpoint.

    Args:
        name: Human-readable checkpoint name.
        system_prompt: The system prompt text.
        messages: Full message list (excluding system prompt).
        model: Active model name.
        token_count: Current token count.
        project_dir: Project root directory.

    Returns:
        Path to the saved checkpoint file.
    """
    checkpoint_dir = _checkpoints_dir(project_dir)
    # Sanitize name for filesystem
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    path = checkpoint_dir / f"{safe_name}.checkpoint.json"

    data = {
        "name": name,
        "timestamp": time.time(),
        "model": model,
        "token_count": token_count,
        "system_prompt": system_prompt,
        "messages": messages,
    }

    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    logger.info("Checkpoint saved name=%s path=%s tokens=%d", name, path, token_count)
    return path


def load_checkpoint(
    name: str,
    project_dir: Path,
) -> dict[str, Any] | None:
    """Load a checkpoint by name.

    Returns the checkpoint data dict, or None if not found.
    """
    checkpoint_dir = _checkpoints_dir(project_dir)
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    path = checkpoint_dir / f"{safe_name}.checkpoint.json"

    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        logger.info("Checkpoint loaded name=%s tokens=%d", name, data.get("token_count", 0))
        return data
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Failed to load checkpoint name=%s error=%s", name, exc)
        return None


def list_checkpoints(project_dir: Path) -> list[dict[str, Any]]:
    """List available checkpoints with metadata.

    Returns a list of dicts with name, timestamp, model, token_count.
    """
    checkpoint_dir = _checkpoints_dir(project_dir)
    results: list[dict[str, Any]] = []

    for path in sorted(checkpoint_dir.glob("*.checkpoint.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            results.append(
                {
                    "name": data.get("name", path.stem),
                    "timestamp": data.get("timestamp", 0),
                    "model": data.get("model", "unknown"),
                    "token_count": data.get("token_count", 0),
                    "message_count": len(data.get("messages", [])),
                }
            )
        except (json.JSONDecodeError, OSError):
            continue

    return results


def delete_checkpoint(name: str, project_dir: Path) -> bool:
    """Delete a checkpoint by name. Returns True if deleted."""
    checkpoint_dir = _checkpoints_dir(project_dir)
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    path = checkpoint_dir / f"{safe_name}.checkpoint.json"

    if path.exists():
        path.unlink()
        logger.info("Checkpoint deleted name=%s", name)
        return True
    return False
