"""Stage uncommitted but git-added changes ready for review + commit."""

from __future__ import annotations

import subprocess
from pathlib import Path


def _run(*args: str) -> None:
    subprocess.run(args, check=True, capture_output=True)


_run("git", "init", "-q")
_run("git", "config", "user.email", "bench@example.com")
_run("git", "config", "user.name", "Benchmark")

Path("handler.py").write_text("def handler(event):\n    return {'ok': True}\n", encoding="utf-8")
_run("git", "add", ".")
_run("git", "commit", "-q", "-m", "baseline handler")

# Meaningful change to review and commit.
Path("handler.py").write_text(
    "def handler(event):\n"
    "    if not event:\n"
    "        return {'ok': False, 'error': 'empty event'}\n"
    "    return {'ok': True, 'event_id': event.get('id')}\n",
    encoding="utf-8",
)
_run("git", "add", "handler.py")
