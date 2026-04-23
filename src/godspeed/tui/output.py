"""Rich output formatting for the Godspeed TUI — Midnight Gold theme.

Design philosophy: function first with beautiful form. Whitespace as structure,
restraint in color, information density that respects the developer's attention.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from godspeed.tui import safe_console as _tsc
from godspeed.tui.theme import (
    BOLD_ERROR,
    BOLD_PRIMARY,
    BOLD_WARNING,
    BRAND_TAGLINE,
    DECORATOR,
    DIM,
    ERROR,
    GUTTER,
    GUTTER_STYLE,
    MARKER_ERROR,
    MARKER_INFO,
    MARKER_PARALLEL,
    MARKER_SUCCESS,
    MARKER_TOOL,
    MARKER_WARNING,
    MUTED,
    PROMPT_ICON,
    RULE_CHAR,
    SECONDARY,
    SEPARATOR_DOT,
    SUCCESS,
    SYNTAX_THEME,
    TABLE_KEY,
    TABLE_VALUE,
    WARNING,
    brand,
    styled,
    styled_escaped,
)

logger = logging.getLogger(__name__)

console = Console()


def escape_markup(text: object) -> str:
    """Escape *text* for embedding in Rich markup."""
    return _tsc.escape_markup(text)


def print_markup_safe(line: str, *, highlight: bool = False, end: str = "\n") -> None:
    """Print a markup line using the shared TUI console; never raise ``MarkupError``."""
    _tsc.print_markup_safe(console, line, highlight=highlight, end=end)


def print_plain_safe(*objects: Any, sep: str = " ", end: str = "\n", **kwargs: Any) -> None:
    """Print without interpreting ``[...]`` as Rich markup (model/shell/plain output)."""
    _tsc.print_plain_safe(console, *objects, sep=sep, end=end, **kwargs)


# Max lines for inline tool result display
_RESULT_MAX_LINES = 10
_RESULT_MAX_CHARS = 2000

# Rule width for horizontal separators
_RULE_WIDTH = 35


def _rule() -> str:
    """Return a thin horizontal rule string."""
    return styled(RULE_CHAR * _RULE_WIDTH, MUTED)


def _safe_print_markup(line: str) -> None:
    """Print a line that may include Rich markup; never raise on unbalanced or hostile input."""
    print_markup_safe(line, highlight=False)


def _gutter_lines(text: str) -> None:
    """Print text with a left gutter border on each line."""
    gutter = styled(GUTTER, GUTTER_STYLE)
    for line in text.splitlines():
        print_markup_safe(f"    {gutter} {escape(line)}")


# =============================================================================
# Status-typed message formatters
# =============================================================================


def format_info(message: str) -> None:
    """Display an info message with ● indicator."""
    _safe_print_markup(f"  {styled(MARKER_INFO, SECONDARY)} {styled(message, DIM)}")


def format_success(message: str) -> None:
    """Display a success message with ✓ indicator."""
    _safe_print_markup(f"  {styled(MARKER_SUCCESS, SUCCESS)} {message}")


def format_warning(message: str) -> None:
    """Display a warning message with ⚠ indicator."""
    _safe_print_markup(f"  {styled(MARKER_WARNING, WARNING)} {message}")


def format_error(message: str) -> None:
    """Display an error message with ✗ indicator."""
    _safe_print_markup(
        f"  {styled(MARKER_ERROR, ERROR)} {styled_escaped('Error: ' + message, BOLD_ERROR)}"
    )


# =============================================================================
# Thinking display
# =============================================================================


def format_thinking(text: str) -> None:
    """Display extended thinking content in a dim collapsible-style panel."""
    if not text.strip():
        return
    # Truncate very long thinking for display (keep first 2000 chars)
    display_text = text[:2000]
    if len(text) > 2000:
        display_text += f"\n... ({len(text) - 2000} chars truncated)"

    panel = Panel(
        styled_escaped(display_text, DIM),
        title=styled("Thinking", MUTED),
        border_style=MUTED,
        expand=False,
        padding=(0, 1),
    )
    console.print(panel)


# =============================================================================
# Tool call / result display
# =============================================================================


def format_tool_call(name: str, args: dict[str, Any]) -> None:
    """Display a tool call — compact inline for simple calls, expanded for complex ones."""
    marker = styled(MARKER_TOOL, MUTED)
    tool = styled(name, BOLD_PRIMARY)

    # Simple single-arg tools: compact inline
    if name in ("file_read", "glob_search", "repo_map") and args.get("file_path"):
        print_markup_safe(f"  {marker} {tool}  {escape(str(args['file_path']))}")
        return

    if name == "grep_search" and args.get("pattern"):
        path = args.get("path", "")
        suffix = f"  {escape(path)}" if path else ""
        print_markup_safe(f"  {marker} {tool}  {styled_escaped(str(args['pattern']), DIM)}{suffix}")
        return

    if name == "git" and args.get("action"):
        action = str(args["action"])
        extra = args.get("message", args.get("branch", ""))
        suffix = f"  {escape(str(extra))}" if extra else ""
        print_markup_safe(f"  {marker} {tool}  {escape(action)}{suffix}")
        return

    # Shell: show command with $ prefix and gutter
    if name == "shell" and args.get("command"):
        print_markup_safe(f"  {marker} {tool}")
        _gutter_lines(f"$ {args['command']}")
        return

    # File edit: show compact diff with gutter
    if name == "file_edit" and args.get("file_path"):
        print_markup_safe(f"  {marker} {tool}  {escape(str(args['file_path']))}")
        if args.get("old_string") and args.get("new_string"):
            import difflib

            old_lines = args["old_string"].splitlines()
            new_lines = args["new_string"].splitlines()
            diff_output = list(difflib.unified_diff(old_lines, new_lines, lineterm="", n=1))
            # Skip --- / +++ headers
            diff_lines = diff_output[2:] if len(diff_output) > 2 else diff_output
            if diff_lines:
                diff_text = "\n".join(diff_lines[:15])
                _gutter_lines(diff_text)
        return

    # File write: show path and line count
    if name == "file_write" and args.get("file_path"):
        content = args.get("content", "")
        line_count = len(content.splitlines())
        count_label = styled(f"({line_count} lines)", DIM)
        print_markup_safe(f"  {marker} {tool}  {escape(str(args['file_path']))}  {count_label}")
        return

    # Default: JSON args with gutter
    try:
        args_text = json.dumps(args, indent=2, default=str)
    except (TypeError, ValueError):
        args_text = str(args)

    print_markup_safe(f"  {marker} {tool}")
    _gutter_lines(args_text)


def format_tool_result(name: str, result: str, is_error: bool = False) -> None:
    """Display a tool result — compact for success, expanded for errors."""
    if is_error:
        marker = styled(MARKER_ERROR, ERROR)
        tool = styled_escaped(f"{name}", BOLD_ERROR)

        # Show full error output
        display = result
        if len(result) > _RESULT_MAX_CHARS:
            remaining = len(result) - _RESULT_MAX_CHARS
            display = result[:_RESULT_MAX_CHARS] + f"\n... ({remaining} more chars)"

        lines = display.splitlines()
        if len(lines) <= 3:
            # Short error inline
            first = lines[0] if lines else ""
            print_markup_safe(f"  {marker} {tool}  {escape(first)}")
            for line in lines[1:]:
                print_markup_safe(f"    {escape(line)}")
        else:
            print_markup_safe(f"  {marker} {tool}")
            # Indent error output
            for line in lines[:20]:
                print_markup_safe(f"    {styled_escaped(line, DIM)}")
            if len(lines) > 20:
                print_markup_safe(
                    f"    {styled_escaped(f'... ({len(lines) - 20} more lines)', DIM)}"
                )
    else:
        marker = styled(MARKER_SUCCESS, SUCCESS)
        tool = styled(name, MUTED)

        # Summarize success output
        lines = result.splitlines()
        line_count = len(lines)

        if not result.strip():
            print_markup_safe(f"  {marker} {tool}")
            return

        # Short results: show inline
        if line_count <= _RESULT_MAX_LINES and len(result) <= 500:
            print_markup_safe(f"  {marker} {tool}")
            for line in lines:
                print_markup_safe(f"    {styled_escaped(line, DIM)}")
        else:
            # Long results: show summary
            print_markup_safe(f"  {marker} {tool}  {styled(f'({line_count} lines)', DIM)}")


# =============================================================================
# Parallel tool call display
# =============================================================================

# Max characters for inline argument preview in parallel header
_PARALLEL_ARG_MAX = 40


def _tool_brief(name: str, args: dict[str, Any]) -> str:
    """Return a short one-line summary of a tool call for parallel headers."""
    primary = (
        args.get("file_path")
        or args.get("command")
        or args.get("pattern")
        or args.get("action")
        or ""
    )
    if primary and len(primary) > _PARALLEL_ARG_MAX:
        primary = "..." + primary[-(_PARALLEL_ARG_MAX - 3) :]
    if primary:
        return f"{name} {styled_escaped(str(primary), DIM)}"
    return name


def format_parallel_tool_calls(calls: list[tuple[str, dict[str, Any]]]) -> None:
    """Display a grouped header for parallel tool dispatch.

    Shows count and brief tool names so the user knows what is running
    concurrently before results arrive.
    """
    count = len(calls)
    marker = styled(MARKER_PARALLEL, SECONDARY)
    header = styled(f"Running {count} tools in parallel", BOLD_PRIMARY)
    print_markup_safe(f"\n  {marker} {header}")

    for name, args in calls:
        brief = _tool_brief(name, args)
        print_markup_safe(f"    {styled(MARKER_TOOL, MUTED)} {brief}")

    console.print()


def format_parallel_results(results: list[tuple[str, str, bool]]) -> None:
    """Display a batch summary of parallel tool results.

    Each entry is (tool_name, output_preview, is_error). Successes get a
    compact one-liner; errors get the full treatment.
    """
    successes = [(n, o) for n, o, err in results if not err]
    errors = [(n, o) for n, o, err in results if err]

    # Compact success summary
    if successes:
        names = [styled_escaped(n, MUTED) for n, _ in successes]
        label = f" {SEPARATOR_DOT} ".join(names)
        print_markup_safe(f"  {styled(MARKER_SUCCESS, SUCCESS)} {label}")

    # Expanded error display
    for name, output in errors:
        marker = styled(MARKER_ERROR, ERROR)
        tool = styled_escaped(name, BOLD_ERROR)
        preview = output.splitlines()[0] if output.strip() else "(no output)"
        if len(preview) > 120:
            preview = preview[:117] + "..."
        print_markup_safe(f"  {marker} {tool}  {escape(preview)}")

    console.print()


def format_assistant_text(text: str) -> None:
    """Render assistant text as Rich Markdown.

    If Markdown is malformed, fall back to escaped plain text so a bad
    model response cannot break the TUI.
    """
    if not text.strip():
        return
    try:
        console.print(Markdown(text))
    except Exception as exc:
        logger.debug("Assistant Markdown render failed: %s", exc, exc_info=True)
        print_plain_safe(escape(text))


def format_permission_prompt(
    tool_name: str,
    reason: str,
    arguments: dict[str, Any] | None = None,
) -> str:
    """Display a permission request with contextual detail.

    Shows tool-specific previews:
    - file_edit: unified diff of old_string -> new_string
    - file_write: first 15 lines of content
    - shell: syntax-highlighted command
    - file_read / grep_search / repo_map: file path
    """
    console.print()

    # Warning marker header
    warn_icon = styled(MARKER_WARNING, WARNING)
    warn_text = styled("Permission required", BOLD_WARNING)
    print_markup_safe(f"  {warn_icon}  {warn_text}")
    console.print()

    args = arguments or {}

    # Tool name and primary arg
    print_markup_safe(f"    {styled_escaped(str(tool_name), BOLD_PRIMARY)}", end="")

    if tool_name == "file_edit" and args.get("old_string") and args.get("new_string"):
        import difflib

        file_path = args.get("file_path", "unknown")
        old = args["old_string"]
        new = args["new_string"]

        # Line change stats
        old_lines = old.splitlines()
        new_lines = new.splitlines()
        added = sum(1 for line in difflib.ndiff(old_lines, new_lines) if line.startswith("+ "))
        removed = sum(1 for line in difflib.ndiff(old_lines, new_lines) if line.startswith("- "))
        stats = f"+{added} -{removed} lines"

        print_markup_safe(f"  {escape(str(file_path))}  {styled(stats, DIM)}")

        # Unified diff
        diff_output = list(
            difflib.unified_diff(
                old_lines,
                new_lines,
                fromfile="before",
                tofile="after",
                lineterm="",
                n=2,
            )
        )
        diff_content = diff_output[2:] if len(diff_output) > 2 else diff_output
        diff_text = "\n".join(diff_content[:30])
        console.print(Syntax(diff_text, "diff", theme=SYNTAX_THEME, word_wrap=True))
        if len(diff_content) > 30:
            print_markup_safe(f"    {styled(f'... ({len(diff_content) - 30} more lines)', DIM)}")

    elif tool_name == "file_write" and args.get("content"):
        from pathlib import Path

        file_path = args.get("file_path", "unknown")
        all_lines = args["content"].splitlines()
        line_count = len(all_lines)

        target = Path(file_path)
        if not target.is_absolute():
            action = "write"
        elif target.exists():
            action = "overwrite"
        else:
            action = "create"
        action_style = WARNING if action == "overwrite" else MUTED
        print_markup_safe(
            f"  {escape(str(file_path))}  {styled(f'({action}, {line_count} lines)', action_style)}"
        )

        preview = "\n".join(all_lines[:15])
        ext = file_path.rsplit(".", 1)[-1] if "." in file_path else "text"
        lexer_map = {"py": "python", "js": "javascript", "ts": "typescript", "yaml": "yaml"}
        lexer = lexer_map.get(ext, ext)
        console.print(Syntax(preview, lexer, theme=SYNTAX_THEME, word_wrap=True))
        if len(all_lines) > 15:
            print_markup_safe(f"    {styled(f'... ({len(all_lines) - 15} more lines)', DIM)}")

    elif tool_name == "shell" and args.get("command"):
        console.print()
        cmd = f"    $ {args['command']}"
        console.print(Syntax(cmd, "bash", theme=SYNTAX_THEME, word_wrap=True))

    elif args.get("file_path"):
        print_markup_safe(f"  {escape(str(args['file_path']))}")

    elif args.get("pattern"):
        print_markup_safe(f"  {styled_escaped(str(args['pattern']), DIM)}")

    else:
        console.print()

    # Prompt line
    console.print()
    print_markup_safe(f"    {styled_escaped(str(reason), DIM)}")
    print_markup_safe(
        f"    {styled('Allow?', WARNING)}"
        f" {styled(f'(y)es {SEPARATOR_DOT} (n)o {SEPARATOR_DOT} (a)lways this session', DIM)}"
    )
    return ""


def format_permission_denied(tool_name: str, reason: str) -> None:
    """Display a permission denied notice."""
    marker = styled(MARKER_ERROR, ERROR)
    print_markup_safe(
        f"  {marker} {styled('Blocked:', BOLD_ERROR)} "
        f"{escape(str(tool_name))} -- {escape(str(reason))}"
    )


def format_status_hud(
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
    model: str,
    turns: int,
    budget_usd: float = 0.0,
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
) -> None:
    """Print a compact one-line session HUD after each completed turn.

    Example rendering:
        · 1,234 in + 567 out (1,801) · 85% cached · $0.0024 · claude-opus-4-7 · 3 turns

    When ``budget_usd`` > 0, the cost is shown as ``$X / $Y`` with a red
    tint if we're within 20% of the hard limit.

    When ``cache_read_tokens > 0`` (Anthropic-family providers with prompt
    caching enabled), the cache-hit ratio is inlined between tokens and
    cost so users can see real caching ROI after each turn. Callers pass
    the already-known ``LLMClient`` totals so this function stays pure.
    """
    total_tokens = input_tokens + output_tokens
    tokens_text = styled(f"{input_tokens:,} in + {output_tokens:,} out ({total_tokens:,})", DIM)

    # Cache summary — only render when the provider reported cache hits,
    # otherwise it's just noise. Ratio is against total input tokens which
    # already include the cached prefix (per Anthropic accounting).
    cache_text: str | None = None
    if cache_read_tokens > 0 and input_tokens > 0:
        pct = int(100 * cache_read_tokens / input_tokens)
        cache_text = styled(f"{pct}% cached", DIM)
    elif cache_creation_tokens > 0:
        # First-turn cache write — no reads yet, but show the write so
        # the user knows future turns will amortize.
        cache_text = styled(f"{cache_creation_tokens:,} cached", DIM)

    if budget_usd > 0:
        remaining = max(0.0, budget_usd - cost_usd)
        near_limit = remaining < budget_usd * 0.2
        cost_style = WARNING if near_limit else DIM
        cost_text = styled(f"${cost_usd:.4f} / ${budget_usd:.2f}", cost_style)
    else:
        cost_text = styled(f"${cost_usd:.4f}", DIM)

    # Short model label — drop provider prefix for readability when present
    model_short = model.split("/", 1)[-1] if "/" in model else model
    model_text = styled_escaped(str(model_short), MUTED)

    turns_text = styled(f"{turns} turn{'s' if turns != 1 else ''}", DIM)
    sep = styled(SEPARATOR_DOT, MUTED)

    fields: list[str] = [tokens_text]
    if cache_text is not None:
        fields.append(cache_text)
    fields.extend([cost_text, model_text, turns_text])
    print_markup_safe(f"  {sep} " + f" {sep} ".join(fields))


def format_diff_review_prompt(
    tool_name: str,
    path: str,
    before: str,
    after: str,
) -> None:
    """Render a pending-edit diff and prompt text.

    Called by the TUI's DiffReviewer just before the write. Distinct from
    ``format_permission_prompt``: that one asks whether the tool should run
    at all; this one asks whether THIS specific diff should be applied.

    For ``diff_apply`` the ``after`` is the raw unified diff (``before`` is
    empty) — we render it verbatim via the ``diff`` syntax lexer so the
    user sees exactly what will be applied.
    """
    import difflib

    console.print()
    marker = styled(MARKER_WARNING, WARNING)
    header = styled("Review proposed edit", BOLD_WARNING)
    print_markup_safe(f"  {marker}  {header}")
    console.print()
    print_markup_safe(f"    {styled_escaped(str(tool_name), BOLD_PRIMARY)}  {escape(str(path))}")

    # For diff_apply: "before" is empty, "after" is the raw unified diff text.
    if (not before and after.startswith("diff --git")) or after.startswith("---"):
        diff_text = after
    else:
        before_lines = before.splitlines()
        after_lines = after.splitlines()
        added = sum(1 for line in difflib.ndiff(before_lines, after_lines) if line.startswith("+ "))
        removed = sum(
            1 for line in difflib.ndiff(before_lines, after_lines) if line.startswith("- ")
        )
        print_markup_safe(f"    {styled(f'+{added} -{removed} lines', DIM)}")
        diff_lines = list(
            difflib.unified_diff(
                before_lines,
                after_lines,
                fromfile="before",
                tofile="after",
                lineterm="",
                n=3,
            )
        )
        diff_content = diff_lines[2:] if len(diff_lines) > 2 else diff_lines
        diff_text = "\n".join(diff_content[:80])
        if len(diff_content) > 80:
            diff_text += f"\n... ({len(diff_content) - 80} more lines)"

    console.print(Syntax(diff_text, "diff", theme=SYNTAX_THEME, word_wrap=True))
    print_markup_safe(
        f"    {styled('Apply?', WARNING)} {styled(f'(y)es {SEPARATOR_DOT} (n)o', DIM)}"
    )


def format_stats(
    input_tokens: int,
    output_tokens: int,
    model: str,
    session_id: str,
    cost: float | None = None,
) -> None:
    """Display session statistics (used by /stats command)."""
    table = Table(show_header=False, border_style=MUTED, expand=False, padding=(0, 2))
    table.add_column("Key", style=TABLE_KEY)
    table.add_column("Value", style=TABLE_VALUE)
    table.add_row("Model", model)
    table.add_row("Session", session_id[:12] + "...")
    table.add_row("Input tokens", f"{input_tokens:,}")
    table.add_row("Output tokens", f"{output_tokens:,}")
    table.add_row("Total tokens", f"{input_tokens + output_tokens:,}")
    if cost is not None:
        table.add_row("Estimated cost", f"${cost:.4f}")

    panel = Panel(
        table,
        title=styled("Session Stats", BOLD_PRIMARY),
        border_style=SECONDARY,
        expand=False,
    )
    console.print(panel)


def format_welcome(
    model: str,
    project_dir: str,
    tools: list[str] | None = None,
    deny_rules: list[str] | None = None,
    audit_enabled: bool = True,
) -> None:
    """Display welcome banner — clean, minimal, function-first."""
    from godspeed import __version__

    console.print()

    # Decorated branded header
    dec = styled(f"{DECORATOR}{DECORATOR}{DECORATOR}", MUTED)
    header = f"  {dec} {PROMPT_ICON} {brand(__version__)} {dec}"
    print_markup_safe(header)
    print_markup_safe(f"  {styled(BRAND_TAGLINE, DIM)}")
    console.print()

    # Thin rule separator
    print_markup_safe(f"  {_rule()}")

    # Key info — aligned, clean
    audit_status = styled("enabled", SUCCESS) if audit_enabled else styled("disabled", ERROR)
    print_markup_safe(f"  {styled('Model', MUTED)}    {escape(str(model))}")
    print_markup_safe(f"  {styled('Project', MUTED)}  {escape(str(project_dir))}")
    print_markup_safe(f"  {styled('Audit', MUTED)}    {audit_status}")

    # Hint line
    print_markup_safe(
        f"\n  {styled(f'Type /help for commands {SEPARATOR_DOT} /plan for read-only mode', DIM)}\n"
    )


def format_session_summary(
    duration_secs: float,
    input_tokens: int,
    output_tokens: int,
    cost: float | None = None,
    tool_calls: int = 0,
    tool_errors: int = 0,
    tool_denied: int = 0,
) -> None:
    """Display session summary on quit — clean, compact."""
    console.print()
    print_markup_safe(f"  {_rule()}")
    print_markup_safe(f"  {styled('Session complete', DIM)}")
    console.print()

    # Duration
    minutes = int(duration_secs // 60)
    seconds = int(duration_secs % 60)
    dur = f"{minutes}m {seconds}s" if minutes > 0 else f"{seconds}s"
    print_markup_safe(f"    {styled('Duration', MUTED)}  {dur}")

    # Tokens
    total = input_tokens + output_tokens
    print_markup_safe(
        f"    {styled('Tokens', MUTED)}    {total:,}"
        f"  {styled(f'(in: {input_tokens:,} {SEPARATOR_DOT} out: {output_tokens:,})', DIM)}"
    )

    # Cost
    if cost is not None and cost > 0:
        print_markup_safe(f"    {styled('Cost', MUTED)}      ${cost:.4f}")
    elif cost is not None:
        print_markup_safe(f"    {styled('Cost', MUTED)}      {styled('free', SUCCESS)}")

    # Tool summary
    if tool_calls > 0:
        success = tool_calls - tool_errors - tool_denied
        parts = [f"{success} {MARKER_SUCCESS}"]
        if tool_errors > 0:
            parts.append(f"{tool_errors} {MARKER_ERROR}")
        if tool_denied > 0:
            parts.append(f"{tool_denied} denied")
        summary = f" {SEPARATOR_DOT} ".join(parts)
        print_markup_safe(
            f"    {styled('Tools', MUTED)}     {tool_calls} calls  {styled(f'({summary})', DIM)}"
        )

    # Branded sign-off with decorative slashes
    dec = styled(f"{DECORATOR}{DECORATOR}{DECORATOR}", MUTED)
    print_markup_safe(f"\n  {dec} {PROMPT_ICON} {styled('Godspeed', BOLD_PRIMARY)} {dec}\n")
