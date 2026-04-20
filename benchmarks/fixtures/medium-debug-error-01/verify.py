"""get_user must handle missing user_id without raising KeyError."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

src = Path("api/users.py")
if not src.is_file():
    print("api/users.py missing")
    sys.exit(1)

spec = importlib.util.spec_from_file_location("users_mod", src)
mod = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(mod)
except Exception as exc:
    print(f"failed to import: {exc}")
    sys.exit(1)

try:
    result = mod.get_user({})
except KeyError:
    print("KeyError still raised for missing user_id")
    sys.exit(1)
except Exception as exc:
    print(f"unexpected exception: {type(exc).__name__}: {exc}")
    sys.exit(1)

if not isinstance(result, dict):
    print(f"expected dict, got {type(result).__name__}")
    sys.exit(1)
