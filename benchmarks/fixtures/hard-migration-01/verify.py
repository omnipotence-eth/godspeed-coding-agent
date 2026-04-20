"""No file should still import requests; httpx should be used instead."""

from __future__ import annotations

import re
import sys
from pathlib import Path

for py in Path(".").rglob("*.py"):
    if py.name == "verify.py":
        continue
    text = py.read_text(encoding="utf-8")
    if re.search(r"^\s*import\s+requests\b", text, re.MULTILINE):
        print(f"{py} still imports requests")
        sys.exit(1)
    if re.search(r"^\s*from\s+requests\b", text, re.MULTILINE):
        print(f"{py} still imports requests via from")
        sys.exit(1)

if not any(
    "httpx" in p.read_text(encoding="utf-8")
    for p in Path(".").rglob("*.py")
    if p.name != "verify.py"
):
    print("no file references httpx")
    sys.exit(1)
