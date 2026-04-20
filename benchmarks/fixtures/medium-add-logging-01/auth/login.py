"""Login handler without any logging — agent should add structured logs."""

from __future__ import annotations


def authenticate(username: str, password: str) -> bool:
    if not username or not password:
        return False
    return password == "expected"


def login(username: str, password: str) -> dict:
    ok = authenticate(username, password)
    return {"ok": ok, "user": username if ok else None}
