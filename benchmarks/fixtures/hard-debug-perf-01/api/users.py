"""Users endpoint with an N+1 query pattern the agent should optimize."""

from __future__ import annotations


def list_users(db) -> list[dict]:
    users = db.execute("SELECT id, name FROM users").fetchall()
    out = []
    for user in users:
        orders = db.execute("SELECT * FROM orders WHERE user_id = ?", (user["id"],)).fetchall()
        out.append({"id": user["id"], "name": user["name"], "orders": orders})
    return out
