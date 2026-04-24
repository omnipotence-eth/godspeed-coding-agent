"""Project instructions loader — GODSPEED.md, AGENTS.md, CLAUDE.md, .cursorrules, .godspeedrules.

Walks up the directory tree loading applicable instruction files.
Supports the cross-agent AGENTS.md standard (Linux Foundation AAIF) and
reads CLAUDE.md / .cursorrules for zero-friction migration from other agents.

Priority: GODSPEED.md > AGENTS.md > .godspeedrules > CLAUDE.md > .cursorrules
All found files are merged (parent-first, most-specific-last).
"""

# Cross-agent instruction files, in priority order.
# GODSPEED.md is always loaded first and takes highest priority.
# Others are loaded as supplementary context if GODSPEED.md is absent
# or to capture project-level conventions from other agent configs.
INSTRUCTION_FILES = (
    "GODSPEED.md",
    "AGENTS.md",
    "SKILL.md",
    ".godspeedrules",
    "CLAUDE.md",
    ".cursorrules",
)

DEFAULT_FILENAME = "GODSPEED.md"


def _load_single_file(
    cwd: Path,
    filename: str,
    walk_parents: bool = True,
) -> list[tuple[Path, str]]:
    """Load a single instruction filename from cwd upward.

    Returns list of (path, content) tuples found, child-first order.
    """
    found: list[tuple[Path, str]] = []
    current = cwd.resolve()
    root = current.anchor

    while True:
        instructions_path = current / filename
        if instructions_path.is_file():
            try:
                content = instructions_path.read_text(encoding="utf-8").strip()
                if content:
                    found.append((instructions_path, content))
                    logger.info("Found project instructions at %s", instructions_path)
            except OSError as exc:
                logger.warning(
                    "Failed to read instructions %s: %s",
                    instructions_path,
                    exc,
                )

        if not walk_parents or str(current) == root:
            break
        parent = current.parent
        if parent == current:
            break
        current = parent

    # Reverse so parent instructions come first (most general → most specific)
    found.reverse()
    return found


def load_project_instructions(
    cwd: Path,
    filename: str = DEFAULT_FILENAME,
    walk_parents: bool = True,
) -> str | None:
    """Load project instructions from GODSPEED.md and cross-agent config files.

    Searches from cwd upward through parent directories. Loads:
    1. GODSPEED.md (primary — always loaded)
    2. AGENTS.md (Linux Foundation standard)
    3. CLAUDE.md (Claude Code format)
    4. .cursorrules (Cursor format)

    If filename is explicitly set to something other than the default,
    only that filename is loaded (backward-compatible behavior).

    Files are concatenated (parent first, child last) with source headers.

    Args:
        cwd: Starting directory to search from.
        filename: Instruction file name (default: GODSPEED.md).
        walk_parents: Whether to walk up the directory tree.

    Returns:
        Combined instructions text, or None if no files found.
    """
    # If caller specified a non-default filename, use single-file mode
    if filename != DEFAULT_FILENAME:
        found_files = _load_single_file(cwd, filename, walk_parents)
        return _merge_found_files(found_files)

    # Multi-file mode: load all recognized instruction files
    all_found: list[tuple[Path, str]] = []
    seen_paths: set[Path] = set()

    for instruction_file in INSTRUCTION_FILES:
        found = _load_single_file(cwd, instruction_file, walk_parents)
        for path, content in found:
            if path not in seen_paths:
                seen_paths.add(path)
                all_found.append((path, content))

    return _merge_found_files(all_found)


def _merge_found_files(found_files: list[tuple[Path, str]]) -> str | None:
    """Merge found instruction files into a single string."""
    if not found_files:
        return None

    if len(found_files) == 1:
        return found_files[0][1]

    parts = []
    for path, content in found_files:
        parts.append(f"# From: {path}\n\n{content}")
    return "\n\n---\n\n".join(parts)


def find_project_root(cwd: Path, markers: tuple[str, ...] | None = None) -> Path:
    """Find the project root by looking for common markers.

    Args:
        cwd: Starting directory.
        markers: Files/dirs that indicate project root.

    Returns:
        Project root path, or cwd if no markers found.
    """
    if markers is None:
        markers = (
            ".git",
            "pyproject.toml",
            "package.json",
            "Cargo.toml",
            "go.mod",
            ".godspeed",
            "GODSPEED.md",
        )

    current = cwd.resolve()
    while True:
        for marker in markers:
            if (current / marker).exists():
                return current
        parent = current.parent
        if parent == current:
            break
        current = parent

    return cwd
