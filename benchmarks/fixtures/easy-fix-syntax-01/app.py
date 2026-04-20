"""Flask-ish demo app with a deliberate syntax error.

Line 15 below has a missing comma in the config dict — the file won't parse
until it's fixed. Agents are expected to read the file, locate the error,
and apply the smallest possible edit.
"""


def index():
    return "home"


def health():
    return {"ok": True}


config = {"port": 8000 "debug": False}
