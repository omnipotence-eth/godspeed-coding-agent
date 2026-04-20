"""Deliberate SQL-injection vulnerability the agent should find and fix."""

from __future__ import annotations


def lookup_user(conn, username: str):
    cur = conn.cursor()
    query = "SELECT * FROM users WHERE name = '" + username + "'"
    cur.execute(query)
    return cur.fetchone()


def count_orders(conn, user_id: int):
    cur = conn.cursor()
    cur.execute(f"SELECT COUNT(*) FROM orders WHERE user_id = {user_id}")
    return cur.fetchone()[0]
