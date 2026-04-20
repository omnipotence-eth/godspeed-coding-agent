"""Canonical tool-name aliases for common LLM hallucinations.

Small open-source models (and even larger ones under token pressure) often
emit plausible-looking but wrong tool names — ``read_file`` instead of
``file_read``, ``grep`` instead of ``grep_search``, ``glob`` instead of
``glob_search``. Rather than letting those calls fail with "unknown tool,"
we rewrite them at parse time to the canonical name.

This is NOT a general synonym system. Each entry here is a known-frequent
hallucination observed in benchmark runs. Adding a new tool to the agent
does NOT require updating this file — only if the new tool has a known
confusion with an established name.

Populated from the Stage A (2026-04-17) benchmark run against
Qwen3-Coder-30B-A3B-Instruct; see ``experiments/qwen3.6-smoke/post_mortem.md``.
"""

from __future__ import annotations

import logging
from types import MappingProxyType

logger = logging.getLogger(__name__)

# Frozen mapping of hallucinated → canonical. Keep alphabetized for audit.
_ALIASES: MappingProxyType[str, str] = MappingProxyType(
    {
        # Hyphen / underscore / word-order variants
        "edit_file": "file_edit",
        "file-edit": "file_edit",
        "file-read": "file_read",
        "file-write": "file_write",
        "read_file": "file_read",
        "write_file": "file_write",
        # Short forms of search tools
        "glob": "glob_search",
        "grep": "grep_search",
        "search": "grep_search",
        "search_code": "code_search",
        # Common misses / camelCase
        "runTests": "test_runner",
        "run_tests": "test_runner",
        "runtests": "test_runner",
        # Git / github confusion
        "git_status": "git",
        # Background / background_check mismatch (rarer, but observed)
        "background": "background_check",
    }
)


def canonicalize_tool_name(name: str) -> str:
    """Return the canonical name for a tool call.

    Rewrites known hallucinations in-place. Unknown names pass through
    unchanged so the registry can reject them with a clear error.
    Logs every rewrite at INFO so users see what happened.
    """
    if not name:
        return name
    canonical = _ALIASES.get(name)
    if canonical is None:
        return name
    logger.info("Tool alias rewrite %r -> %r", name, canonical)
    return canonical
