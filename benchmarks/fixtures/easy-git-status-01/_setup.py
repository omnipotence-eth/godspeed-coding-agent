"""Create a git repo with a committed baseline plus uncommitted edits."""

from __future__ import annotations

import subprocess
from pathlib import Path


def _run(*args: str) -> None:
    subprocess.run(args, check=True, capture_output=True)


_run("git", "init", "-q")
_run("git", "config", "user.email", "bench@example.com")
_run("git", "config", "user.name", "Benchmark")

Path("README.md").write_text("# Demo\n", encoding="utf-8")
Path("main.py").write_text("print('hi')\n", encoding="utf-8")
_run("git", "add", ".")
_run("git", "commit", "-q", "-m", "baseline")

# Dirty state the agent should surface via `git status`.
Path("main.py").write_text("print('hello')\n", encoding="utf-8")
Path("notes.txt").write_text("scratch notes\n", encoding="utf-8")
