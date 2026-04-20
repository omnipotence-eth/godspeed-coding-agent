"""The staged change must be committed (HEAD advances, working tree clean)."""

from __future__ import annotations

import subprocess
import sys

log = subprocess.run(["git", "log", "--oneline"], capture_output=True, text=True, check=False)
if log.returncode != 0:
    print(f"git log failed: {log.stderr}")
    sys.exit(1)

commits = [line for line in log.stdout.splitlines() if line.strip()]
if len(commits) < 2:
    print(f"expected >=2 commits (baseline + review), got {len(commits)}")
    sys.exit(1)

status = subprocess.run(
    ["git", "status", "--porcelain"], capture_output=True, text=True, check=False
)
if status.stdout.strip():
    print(f"working tree still dirty:\n{status.stdout}")
    sys.exit(1)
