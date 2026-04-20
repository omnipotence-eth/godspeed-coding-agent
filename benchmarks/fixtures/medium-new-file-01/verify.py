"""src/utils/validators.py must exist with email and phone validation."""

from __future__ import annotations

import re
import sys
from pathlib import Path

src = Path("src/utils/validators.py")
if not src.is_file():
    print("src/utils/validators.py missing")
    sys.exit(1)

text = src.read_text(encoding="utf-8")
if not re.search(r"def\s+(validate_)?email\w*\s*\(", text):
    print("no email validation function")
    sys.exit(1)
if not re.search(r"def\s+(validate_)?phone\w*\s*\(", text):
    print("no phone validation function")
    sys.exit(1)
