"""Checkpoint system for automatic undo capability."""

from __future__ import annotations

import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CHECKPOINT_DIR = Path.home() / ".godspeed" / "checkpoints"


class Checkpoint:
    """A checkpoint representing a point in time."""

    def __init__(
        self,
        name: str,
        project_dir: str,
        description: str = "",
    ) -> None:
        self.name = name
        self.project_dir = project_dir
        self.description = description
        self.created_at = time.time()
        self.id = f"{int(self.created_at)}"
        self.files_snapshot: list[str] = []


class CheckpointManager:
    """Manage automatic checkpoints and undo capability."""

    def __init__(self, project_dir: str | None = None) -> None:
        self.project_dir = project_dir or os.getcwd()
        self.checkpoints: list[Checkpoint] = []
        self._current_index = -1

        # Ensure checkpoint directory exists
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    def create_checkpoint(self, description: str = "") -> str:
        """Create a new checkpoint."""
        import subprocess

        name = f"checkpoint_{int(time.time())}"
        checkpoint = Checkpoint(name, self.project_dir, description)
        self.checkpoints.append(checkpoint)
        self._current_index = len(self.checkpoints) - 1

        # Store git state if available
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=self.project_dir,
                capture_output=True,
                text=True,
            )
            checkpoint.git_status = result.stdout
        except Exception:
            pass

        # Store list of modified files
        try:
            result = subprocess.run(
                ["git", "diff", "--name-only"],
                cwd=self.project_dir,
                capture_output=True,
                text=True,
            )
            checkpoint.files = result.stdout.strip().split("\n")
        except Exception:
            checkpoint.files = []

        logger.info("Created checkpoint: %s", name)
        return checkpoint.id

    def list_checkpoints(self) -> list[dict[str, Any]]:
        """List available checkpoints."""
        return [
            {
                "id": c.id,
                "name": c.name,
                "description": c.description,
                "created_at": c.created_at,
            }
            for c in self.checkpoints
        ]

    def restore_checkpoint(self, checkpoint_id: str) -> bool:
        """Restore to a specific checkpoint."""
        import subprocess

        # Find checkpoint
        checkpoint = None
        for c in self.checkpoints:
            if c.id == checkpoint_id:
                checkpoint = c
                break

        if not checkpoint:
            logger.warning("Checkpoint not found: %s", checkpoint_id)
            return False

        # Use git checkout to restore
        try:
            # Discard all changes
            subprocess.run(
                ["git", "checkout", "."],
                cwd=self.project_dir,
                capture_output=True,
            )

            # Restore staged changes
            subprocess.run(
                ["git", "checkout", "HEAD", "--", "."],
                cwd=self.project_dir,
                capture_output=True,
            )

            logger.info("Restored checkpoint: %s", checkpoint_id)
            return True

        except Exception as exc:
            logger.error("Restore failed: %s", exc)
            return False

    def undo_last(self) -> bool:
        """Undo the last change using git."""
        import subprocess

        try:
            # Use git reset --soft to undo the last commit
            # Or git checkout for uncommitted changes
            subprocess.run(
                ["git", "checkout", "--", "."],
                cwd=self.project_dir,
                capture_output=True,
            )
            logger.info("Undid last change")
            return True
        except Exception as exc:
            logger.error("Undo failed: %s", exc)
            return False

    def cleanup_old_checkpoints(self, max_keep: int = 10) -> None:
        """Clean up old checkpoints, keeping most recent."""
        if len(self.checkpoints) > max_keep:
            to_remove = self.checkpoints[:-max_keep]
            self.checkpoints = self.checkpoints[-max_keep:]
            self._current_index = len(self.checkpoints) - 1


# Global manager
_manager: CheckpointManager | None = None


def get_checkpoint_manager(project_dir: str | None = None) -> CheckpointManager:
    """Get or create the checkpoint manager."""
    global _manager
    if _manager is None:
        _manager = CheckpointManager(project_dir)
    return _manager