"""Tests for __main__.py module entry point."""

from __future__ import annotations

from unittest.mock import patch

from godspeed.__main__ import main as _  # noqa: F401 — import to verify it exists


def test_main_called_when_run_as_module() -> None:
    with patch("godspeed.__main__.main") as mock_main:
        code = compile(
            "if __name__ == '__main__':\n    main()",
            "godspeed/__main__.py",
            "exec",
        )
        ns: dict[str, object] = {"__name__": "__main__", "main": mock_main}
        exec(code, ns)
        mock_main.assert_called_once()


def test_main_not_called_when_imported() -> None:
    with patch("godspeed.__main__.main") as mock_main:
        code = compile(
            "if __name__ == '__main__':\n    main()",
            "godspeed/__main__.py",
            "exec",
        )
        ns: dict[str, object] = {"__name__": "godspeed.__main__", "main": mock_main}
        exec(code, ns)
        mock_main.assert_not_called()
