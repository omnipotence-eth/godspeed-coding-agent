"""Buggy API handler — accesses request['user_id'] without guarding."""

from __future__ import annotations


def get_user(request: dict) -> dict:
    user_id = request["user_id"]
    return {"id": user_id, "status": "ok"}
