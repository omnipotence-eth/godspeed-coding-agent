"""login.py should import logging and call a logger method."""

from __future__ import annotations

import sys
from pathlib import Path

src = Path("auth/login.py")
if not src.is_file():
    print("auth/login.py missing")
    sys.exit(1)

text = src.read_text(encoding="utf-8")
if "logging" not in text and "logger" not in text.lower():
    print("no logging import/reference added")
    sys.exit(1)
if not any(token in text for token in (".info(", ".warning(", ".error(", ".debug(")):
    print("no logger call added")
    sys.exit(1)
