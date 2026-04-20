"""Every route file must gain some form of error handling."""

from __future__ import annotations

import sys
from pathlib import Path

routes = Path("routes")
files = [f for f in routes.glob("*.py") if f.name != "__init__.py"]
if not files:
    print("no route files found")
    sys.exit(1)

missing: list[str] = []
for f in files:
    text = f.read_text(encoding="utf-8")
    has_handling = any(
        token in text
        for token in ("try:", "except", ".get(", "if ", "raise HTTPException", "abort(")
    )
    if not has_handling:
        missing.append(f.name)

if missing:
    print(f"no error handling added to: {', '.join(missing)}")
    sys.exit(1)
