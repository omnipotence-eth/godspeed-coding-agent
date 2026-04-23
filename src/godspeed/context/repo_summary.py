"""Auto-inject a repo map into the system prompt for large projects.

Aider's biggest edge over naive coding agents is that the model is
told the shape of the codebase up front, so it knows where to look
without having to `glob_search` the whole thing. Godspeed's equivalent:
compute a tree-sitter symbol outline at session start, truncate it to
fit a token budget, and feed it to `build_system_prompt` as the
``repo_map_summary`` argument.

Graceful fallback chain:
1. tree-sitter available + project has ≥ MIN_FILES_FOR_INJECTION
   source files → full symbol outline, truncated to MAX_SUMMARY_CHARS.
2. tree-sitter not installed → simple file-list fallback (still better
   than nothing; lists top-level + one level down).
3. Tiny project (< MIN_FILES_FOR_INJECTION) → return None so
   ``build_system_prompt`` skips the section entirely.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Below this file count, the model doesn't need a map — it can just
# read the few files directly. Injecting noise here costs tokens for
# no benefit.
MIN_FILES_FOR_INJECTION = 10

# Upper bound on the repo-summary section. ~8k chars ≈ 2k tokens —
# small enough to always fit, large enough to meaningfully help.
# Lives inside the cached system-prompt prefix so the cost is paid
# once per session, not per turn.
MAX_SUMMARY_CHARS = 8000

# Directories we never include in the summary fallback — the excludes
# list in tools/excludes.py covers the tree-sitter path, but the
# simple-list fallback needs its own check.
_FALLBACK_SKIP_DIRS = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "dist",
        "build",
        "target",
        ".godspeed",
    }
)

# File extensions we consider "source" when deciding whether a repo
# is big enough to warrant injecting a map.
_SOURCE_EXTS = frozenset(
    {
        ".py",
        ".pyi",
        ".js",
        ".jsx",
        ".ts",
        ".tsx",
        ".go",
        ".rs",
        ".java",
        ".kt",
        ".rb",
        ".c",
        ".cpp",
        ".h",
        ".hpp",
        ".cs",
    }
)


def _count_source_files(project_dir: Path) -> int:
    """Quick O(files) pass to gate whether injection is worthwhile.

    Stops counting at ``MIN_FILES_FOR_INJECTION`` — we only need a
    yes/no answer, not an exact count. Relative cheap on typical
    projects; the tree-sitter pass is the real work.
    """
    count = 0
    for path in project_dir.rglob("*"):
        if not path.is_file():
            continue
        # Skip excluded trees early.
        try:
            rel = path.relative_to(project_dir)
        except ValueError:
            continue
        if any(part in _FALLBACK_SKIP_DIRS for part in rel.parts):
            continue
        if path.suffix.lower() not in _SOURCE_EXTS:
            continue
        count += 1
        if count >= MIN_FILES_FOR_INJECTION:
            return count
    return count


def _fallback_listing(project_dir: Path) -> str:
    """Simple 2-level directory listing when tree-sitter isn't available.

    Better than nothing: gives the model a sense of top-level layout
    so it can `glob_search` / `grep_search` with intent. Capped hard
    to stay inside the char budget.
    """
    lines: list[str] = []
    try:
        entries = sorted(project_dir.iterdir(), key=lambda p: (not p.is_dir(), p.name))
    except OSError:
        return ""
    for entry in entries:
        if entry.name in _FALLBACK_SKIP_DIRS or entry.name.startswith("."):
            continue
        if entry.is_dir():
            lines.append(f"{entry.name}/")
            try:
                children = sorted(entry.iterdir(), key=lambda p: (not p.is_dir(), p.name))
            except OSError:
                continue
            for child in children[:12]:  # cap per-dir to keep summary tight
                if child.name.startswith(".") or child.name in _FALLBACK_SKIP_DIRS:
                    continue
                suffix = "/" if child.is_dir() else ""
                lines.append(f"  {child.name}{suffix}")
            if len(children) > 12:
                lines.append(f"  ... ({len(children) - 12} more)")
        else:
            lines.append(entry.name)
    return "\n".join(lines)


def build_repo_summary(project_dir: Path) -> str | None:
    """Return a repo summary suitable for the system prompt, or ``None``.

    Returns ``None`` when the project is too small to benefit, when
    the project directory doesn't exist, or when all other paths
    produce empty output — signaling :func:`build_system_prompt` to
    skip the section entirely.

    Never raises: any unexpected error during summary construction is
    logged at debug level and treated as "skip the injection."
    """
    if not project_dir.is_dir():
        return None

    try:
        file_count = _count_source_files(project_dir)
    except OSError as exc:
        logger.debug("Repo summary: file count failed: %s", exc)
        return None

    if file_count < MIN_FILES_FOR_INJECTION:
        return None

    # Preferred path: tree-sitter symbol outline.
    summary_text = ""
    try:
        from godspeed.context.repo_map import RepoMapper

        mapper = RepoMapper()
        if mapper.available:
            summary_text = mapper.map_directory(project_dir, max_depth=5)
    except Exception as exc:
        # Broad catch is deliberate — a summary failure must never
        # break session start. Log at debug so verbose users can see.
        logger.debug("Repo summary: tree-sitter path failed: %s", exc)
        summary_text = ""

    # Fallback when tree-sitter is unavailable or produced nothing.
    if not summary_text or summary_text.startswith(("tree-sitter not", "No symbols", "Not a")):
        try:
            summary_text = _fallback_listing(project_dir)
        except OSError as exc:
            logger.debug("Repo summary: fallback listing failed: %s", exc)
            return None

    if not summary_text:
        return None

    if len(summary_text) > MAX_SUMMARY_CHARS:
        truncated = summary_text[:MAX_SUMMARY_CHARS].rsplit("\n", 1)[0]
        summary_text = f"{truncated}\n... (summary truncated at {MAX_SUMMARY_CHARS} chars)"

    return summary_text
