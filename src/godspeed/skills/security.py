"""Skill security scanning — static analysis of skill files before installation."""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

PATTERNS: list[tuple[str, re.Pattern]] = [
    ("obfuscated-eval", re.compile(r"\beval\s*\(", re.I)),
    ("obfuscated-exec", re.compile(r"\bexec\s*\(", re.I)),
    ("base64-exec", re.compile(r"__import__\s*\(\s*['\"]base64['\"]")),
    ("encoded-payload", re.compile(r"[A-Za-z0-9+/]{100,}={0,2}")),
    ("shell-pipe-curl-sh", re.compile(r"curl\s+\S+\s*\|\s*(?:bash|sh)")),
    ("wget-pipe-shell", re.compile(r"wget\s+\S+\s*\|\s*(?:bash|sh)")),
    (
        "hardcoded-token",
        re.compile(r"(?i)(?:api[_-]?key|secret|token)\s*[:=]\s*['\"][A-Za-z0-9_\-]{16,}"),
    ),
    ("crypto-miner", re.compile(r"(?i)(?:minerd|cryptonight|xmrig|ethminer)")),
    ("known-vuln-template", re.compile(r"\{\{.*__globals__.*\}\}")),
    ("env-exfil", re.compile(r"(?i)(?:os\.environ|\$ENV|process\.env)\b")),
    ("ssh-key-inline", re.compile(r"-----BEGIN\s+(?:RSA|DSA|EC|OPENSSH)\s+PRIVATE\s+KEY-----")),
    ("dangerous-shell", re.compile(r"(?:rm\s+-rf\s+[/\\]|mkfs\.|dd\s+if=)")),
]

MAX_FILE_SIZE = 100_000


def scan_skill(skill_dir: Path) -> list[str]:
    """Scan a skill directory for security issues.

    Returns a list of issue descriptions. Empty list = clean.
    """
    issues: list[str] = []

    if not skill_dir.is_dir():
        return ["not-a-directory"]

    skill_path = skill_dir / "SKILL.md"
    if not skill_path.is_file():
        return ["missing-SKILL.md"]

    for fpath in _walk(skill_dir):
        issues.extend(_scan_file(fpath))

    return issues


def _walk(directory: Path) -> list[Path]:
    files = []
    try:
        for f in directory.rglob("*"):
            if f.is_file() and f.name != ".gitkeep":
                files.append(f)
    except OSError:
        pass
    return files


def _scan_file(path: Path) -> list[str]:
    issues: list[str] = []

    try:
        size = path.stat().st_size
    except OSError:
        return ["unreadable"]

    if size > MAX_FILE_SIZE:
        return [f"oversized ({size} bytes, max {MAX_FILE_SIZE})"]

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ["unreadable"]

    for name, pattern in PATTERNS:
        if pattern.search(text):
            issues.append(f"{name} in {path.name}")

    return issues


def classify_risk(issues: list[str]) -> str:
    if not issues:
        return "clean"
    high_risk = {"ssh-key-inline", "crypto-miner", "dangerous-shell", "env-exfil"}
    for i in issues:
        for hr in high_risk:
            if hr in i:
                return "dangerous"
    return "suspicious"
