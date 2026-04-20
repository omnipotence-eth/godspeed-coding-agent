"""A GitHub Actions workflow with lint/type/test stages must exist."""

from __future__ import annotations

import sys
from pathlib import Path

workflows = list(Path(".github/workflows").glob("*.yml")) + list(
    Path(".github/workflows").glob("*.yaml")
)
if not workflows:
    print(".github/workflows/ has no yml/yaml files")
    sys.exit(1)

text = "\n".join(w.read_text(encoding="utf-8").lower() for w in workflows)
required = {
    "lint": ("ruff", "flake8", "pylint", "black", "lint"),
    "type": ("mypy", "ty ", "pyright", "type-check", "typecheck"),
    "test": ("pytest", "unittest", "tests"),
}
missing = [k for k, tokens in required.items() if not any(t in text for t in tokens)]
if missing:
    print(f"workflow missing stages: {', '.join(missing)}")
    sys.exit(1)
