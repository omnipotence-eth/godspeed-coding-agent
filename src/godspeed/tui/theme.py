"""Godspeed theme — Clean minimal like Claude Code / OpenCode.

Design philosophy:
- Minimal: no distractions, pure information  
- Clean monochrome: black/white/grey only
- Text-first: simple characters
- Professional: just what you need
"""

from __future__ import annotations

# =============================================================================
# Core palette — Clean monochrome
# =============================================================================

PRIMARY = "white"
SECONDARY = "grey"  
SUCCESS = "green"
ERROR = "red"
WARNING = "yellow"
MUTED = "dim"
ACCENT = "white"

# Branded
BRAND_GOLD = "white"
BRAND_GOLD_BOLD = "bold white"

# =============================================================================
# Semantic styles  
# =============================================================================

BOLD_PRIMARY = "bold white"
BOLD_SECONDARY = "bold"
BOLD_SUCCESS = "bold green"
BOLD_ERROR = "bold red"
BOLD_WARNING = "bold yellow"
DIM = "dim"

# Panel borders - minimal
BORDER_BRAND = "white"
BORDER_TOOL = "grey"
BORDER_INFO = "grey"
BORDER_SUCCESS = "green"
BORDER_ERROR = "red"
BORDER_WARNING = "yellow"

# Table
TABLE_HEADER = "bold"
TABLE_BORDER = "grey"
TABLE_KEY = "grey"
TABLE_VALUE = "bold"

# Permissions
PERM_ALLOW = "green"
PERM_DENY = "red"
PERM_ASK = "yellow"
PERM_SESSION = "white"

CTX_OK = "green"
CTX_WARN = "yellow"
CTX_CRITICAL = "red"

# =============================================================================
# Branding
# =============================================================================

PROMPT_ICON = ">"  
PROMPT_TEXT = "godspeed"
BRAND_TAGLINE = ""

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
    icon = PROMPT_ICON
    suffix = ""
    if state == "plan":
        suffix = " [plan]"
    elif state == "paused":
        suffix = " [paused]"
    return f"{icon} godspeed{suffix}> "


def format_permission_prompt(tool_name: str, reason: str, arguments: dict) -> None:
    """Display a minimal permission prompt."""
    from godspeed.tui.output import console
    
    console.print()
    console.print(f"  Allow {tool_name}? ({reason})")
    console.print(f"  [y]es / [n]o / [a]lways")
    console.print()


def format_permission_denied(tool_name: str, reason: str) -> None:
    """Display a minimal permission denied notice."""
    from godspeed.tui.output import console
    
    console.print(f"  [{ERROR}]Denied: {tool_name}[/{ERROR}] ({reason})")