"""Code using the requests library that the agent should migrate to httpx."""

from __future__ import annotations

import requests


def fetch_user(user_id: int):
    resp = requests.get(f"https://api.example.com/users/{user_id}")
    resp.raise_for_status()
    return resp.json()


def post_event(payload: dict):
    resp = requests.post("https://api.example.com/events", json=payload)
    return resp.status_code
