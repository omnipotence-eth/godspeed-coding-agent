"""A validate_user_input function must exist and create_user must call it."""

from __future__ import annotations

import re
import sys
from pathlib import Path

src = Path("users.py")
if not src.is_file():
    print("users.py missing")
    sys.exit(1)

text = src.read_text(encoding="utf-8")
if not re.search(r"def\s+validate_user_input\s*\(", text):
    print("validate_user_input function not defined")
    sys.exit(1)
if "validate_user_input(" not in text.replace("def validate_user_input(", "", 1):
    print("create_user does not call validate_user_input")
    sys.exit(1)
