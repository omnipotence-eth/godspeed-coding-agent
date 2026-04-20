"""Fail if app.py still has a syntax error."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

src = Path("app.py")
if not src.is_file():
    print("app.py missing")
    sys.exit(1)

try:
    ast.parse(src.read_text(encoding="utf-8"))
except SyntaxError as exc:
    print(f"syntax error still present: {exc}")
    sys.exit(1)
