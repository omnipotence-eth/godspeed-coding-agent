"""Some test file exercising auth must exist and be non-trivial."""

from __future__ import annotations

import sys
from pathlib import Path

tests = list(Path(".").rglob("test_*.py")) + list(Path(".").rglob("*_test.py"))
tests = [
    t for t in tests if "auth" in t.name.lower() or "auth" in t.read_text(encoding="utf-8").lower()
]
if not tests:
    print("no auth-related test file created")
    sys.exit(1)

text = "\n".join(t.read_text(encoding="utf-8") for t in tests)
if text.count("def test_") < 2:
    print("expected at least 2 test_ functions across generated tests")
    sys.exit(1)
