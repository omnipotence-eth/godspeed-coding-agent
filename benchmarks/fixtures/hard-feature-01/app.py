"""FastAPI skeleton — agent should add rate limiting middleware."""

from __future__ import annotations


class FastAPI:
    def __init__(self):
        self.middleware = []

    def add_middleware(self, mw, **kwargs):
        self.middleware.append((mw, kwargs))

    def get(self, path):
        def deco(fn):
            return fn

        return deco


app = FastAPI()


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/users")
def users():
    return []
