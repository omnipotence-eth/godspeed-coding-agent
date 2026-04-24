"""Godspeed theme — Warm professional like Claude Code.

Design philosophy:
- Warm amber/gold palette (like Claude Code)
- Professional and inviting
- Clean minimal
- Easy on the eyes
"""

from __future__ import annotations

# =============================================================================
# Core palette — Warm amber/gold (Claude Code inspired)
# =============================================================================

PRIMARY = "white"
SECONDARY = "grey"
SUCCESS = "green"
ERROR = "red"
WARNING = "yellow"
MUTED = "dim"
ACCENT = "white"

# Warm amber palette (signature)
AMBER = "yellow"
AMBER_GOLD = "yellow1"
ORANGE = "yellow1"
GOLD_BOLD = "bold yellow1"

# Branded
BRAND_GOLD = "yellow1"
BRAND_GOLD_BOLD = "bold yellow1"

# =============================================================================
# Semantic styles
# =============================================================================

BOLD_PRIMARY = "bold"  # Clean white
BOLD_SECONDARY = "bold"
BOLD_SUCCESS = "bold green"
BOLD_ERROR = "bold red"
BOLD_WARNING = "bold yellow"
DIM = "dim"

# Panel borders
BORDER_BRAND = "yellow1"
BORDER_TOOL = "grey"
BORDER_INFO = "grey"
BORDER_SUCCESS = "green"
BORDER_ERROR = "red"
BORDER_WARNING = "yellow"

# Table
TABLE_HEADER = "bold yellow1"
TABLE_BORDER = "grey"
TABLE_KEY = "grey"
TABLE_VALUE = "bold"

# Permissions
PERM_ALLOW = "green"
PERM_DENY = "red"
PERM_ASK = "yellow1"
PERM_SESSION = "yellow"

CTX_OK = "green"
CTX_WARN = "yellow"
CTX_CRITICAL = "red"

# =============================================================================
# Branding
# =============================================================================

PROMPT_ICON = ">"
PROMPT_TEXT = "godspeed"
BRAND_TAGLINE = "Build fast"

SYNTAX_THEME = "monokai"

# =============================================================================
# Clean markers
# =============================================================================

MARKER_SUCCESS = "✓"
MARKER_ERROR = "✗"
MARKER_WARNING = "!"
MARKER_TOOL = ">"
MARKER_INFO = "i"
MARKER_PARALLEL = "||"
SEPARATOR_DOT = "|"

# Structural
DECORATOR = ""
RULE_CHAR = "-"
GUTTER = ""
GUTTER_STYLE = MUTED

# =============================================================================
# Markup helpers
# =============================================================================


def styled(text: str, style: str) -> str:
    """Wrap text in Rich markup tags."""
    return f"[{style}]{text}[/{style}]"


def brand(version: str = "") -> str:
    """Return the branded product name."""
    name = styled("Godspeed", BOLD_PRIMARY)
    if version:
        return f"{name} {styled(f'v{version}', MUTED)}"
    return name


def icon_prompt(state: str = "") -> str:
    """Return the prompt string."""
    text = "> godspeed"
    if state == "plan":
        text += " [plan]"
    elif state == "paused":
        text += " [paused]"
    text += "> "
    return text


def format_permission_prompt(tool_name: str, reason: str, arguments: dict) -> None:
    """Display a minimal permission prompt."""
    from godspeed.tui.output import console

    console.print()
    console.print(f"  [yellow1]Allow {tool_name}?[/yellow1] ({reason})")
    console.print("  [yellow]y[/yellow]es / [red]n[/red]o / [yellow]a[/yellow]lways")
    console.print()


def format_permission_denied(tool_name: str, reason: str) -> None:
    """Display a minimal permission denied notice."""
    from godspeed.tui.output import console

    console.print(f"  [red]✗ {tool_name}[/red] ({reason})")
