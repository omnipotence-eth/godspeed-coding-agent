"""User creation with inline validation the agent should extract."""

from __future__ import annotations


def create_user(name: str, email: str, age: int) -> dict:
    if not name or len(name) > 100:
        raise ValueError("invalid name")
    if "@" not in email or "." not in email.split("@")[-1]:
        raise ValueError("invalid email")
    if age < 0 or age > 150:
        raise ValueError("invalid age")

    return {"name": name, "email": email, "age": age}
