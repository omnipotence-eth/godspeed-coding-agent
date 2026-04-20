"""Hardcoded DB URL should be gone and replaced with an env-var read."""

from __future__ import annotations

import sys
from pathlib import Path

src = Path("src/db.py")
if not src.is_file():
    print("src/db.py missing")
    sys.exit(1)

text = src.read_text(encoding="utf-8")
if "hunter2@db.internal" in text:
    print("hardcoded credentials still present")
    sys.exit(1)
if "os.environ" not in text and "os.getenv" not in text and "getenv" not in text:
    print("no environment variable read added")
    sys.exit(1)
