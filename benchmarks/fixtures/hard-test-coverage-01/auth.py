"""Auth module without tests."""

from __future__ import annotations


def hash_password(password: str) -> str:
    return f"hashed::{password}"


def verify_password(password: str, hashed: str) -> bool:
    return hashed == hash_password(password)


def token_for(user_id: int) -> str:
    return f"token-{user_id}"
