"""Safe Rich console printing for untrusted model, shell, and path text.

Rich has no ``safe=True`` on ``Console.print``; these helpers approximate it:

* :func:`print_markup_safe` — print a line that may contain Rich markup; on
  :class:`rich.errors.MarkupError`, re-print with escaped content.
* :func:`print_plain_safe` — print with ``markup=False`` and ``highlight=False``.
* :func:`escape_markup` — escape a value for embedding inside trusted markup tags.
"""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.errors import MarkupError
from rich.markup import escape


def escape_markup(text: object) -> str:
    """Escape *text* so it is safe inside a Rich markup string."""
    return escape(str(text))


def print_markup_safe(
    console: Console,
    line: str,
    *,
    highlight: bool = False,
    end: str = "\n",
) -> None:
    """Print a line that may contain Rich markup; never raise ``MarkupError``."""
    try:
        console.print(line, highlight=highlight, end=end)
    except MarkupError:
        console.print(escape(line), highlight=False, markup=True, end=end)


def print_plain_safe(
    console: Console,
    *objects: Any,
    sep: str = " ",
    end: str = "\n",
    **kwargs: Any,
) -> None:
    """Print objects without interpreting ``[...]`` as Rich markup."""
    merged: dict[str, Any] = {**kwargs, "markup": False, "highlight": False}
    console.print(*objects, sep=sep, end=end, **merged)
