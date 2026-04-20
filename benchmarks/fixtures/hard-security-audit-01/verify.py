"""db.py must no longer build SQL via string concat/f-string with inputs."""

from __future__ import annotations

import re
import sys
from pathlib import Path

src = Path("db.py")
if not src.is_file():
    print("db.py missing")
    sys.exit(1)

text = src.read_text(encoding="utf-8")

# Reject any f-string or concatenation forming a SELECT statement.
bad_patterns = [
    re.compile(r'f["\'][^"\']*SELECT', re.IGNORECASE),
    re.compile(r'["\'][^"\']*SELECT[^"\']*["\']\s*\+\s*\w+', re.IGNORECASE),
    re.compile(r'\w+\s*\+\s*["\'][^"\']*SELECT', re.IGNORECASE),
]
for pat in bad_patterns:
    if pat.search(text):
        print(f"unsafe SQL construction still present (pattern {pat.pattern!r})")
        sys.exit(1)

# Require parameterized execute (uses a second argument to .execute).
if not re.search(r"\.execute\s*\([^,]+,\s*[\(\[\{]", text):
    print("no parameterized .execute() call found")
    sys.exit(1)
