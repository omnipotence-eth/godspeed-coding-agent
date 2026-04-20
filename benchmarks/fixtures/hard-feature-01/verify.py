"""Rate limiting must be referenced in the codebase after the agent's changes."""

from __future__ import annotations

import sys
from pathlib import Path

evidence = any(
    any(token in p.read_text(encoding="utf-8").lower() for token in ("ratelimit", "rate_limit", "rate-limit"))
    for p in Path(".").rglob("*.py")
    if p.name != "verify.py"
)
if not evidence:
    print("no rate-limiting code added anywhere")
    sys.exit(1)

# And it must be wired into the app (add_middleware or equivalent).
app = Path("app.py").read_text(encoding="utf-8") if Path("app.py").is_file() else ""
if "add_middleware" not in app and "middleware" not in app.lower():
    print("app.py does not register middleware")
    sys.exit(1)
