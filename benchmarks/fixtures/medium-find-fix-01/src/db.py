"""Database connection with a hardcoded string the agent must move to env."""

from __future__ import annotations

DB_URL = "postgresql://admin:hunter2@db.internal:5432/prod"


def connect():
    return {"url": DB_URL, "connected": True}
