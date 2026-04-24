"""Godspeed visual identity — Ultra-clean minimal.

Design philosophy:
- Minimal: no distractions, pure information
- Fast: load instantly, render lean
- Text-first: characters over emoji when possible
- Professional: clean lines, no clutter
"""

from __future__ import annotations

# =============================================================================
# Core palette — Clean monochrome (terminal-native)
# =============================================================================

PRIMARY = "cyan"  # Cyan — clean, visible
SECONDARY = "grey69"  # Neutral grey
SUCCESS = "green"
ERROR = "red"
WARNING = "yellow"
MUTED = "grey50"  # Dimmed text
ACCENT = "cyan"

BRAND_GOLD = "yellow1"
BRAND_GOLD_BOLD = "bold yellow1"

# =============================================================================
# Semantic styles
# =============================================================================

BOLD_PRIMARY = f"bold {PRIMARY}"
BOLD_SECONDARY = f"bold {SECONDARY}"
BOLD_SUCCESS = f"bold {SUCCESS}"
BOLD_ERROR = f"bold {ERROR}"
BOLD_WARNING = f"bold {WARNING}"
DIM = "dim"

# Panel borders
BORDER_BRAND = "yellow"
BORDER_TOOL = "grey69"
BORDER_INFO = "grey69"
BORDER_SUCCESS = "green"
BORDER_ERROR = "red"
BORDER_WARNING = "yellow"

# Table styles
TABLE_HEADER = "bold cyan"
TABLE_BORDER = "grey46"
TABLE_KEY = "grey35"
TABLE_VALUE = "bold"

PERM_ALLOW = "green"
PERM_DENY = "red"
PERM_ASK = "yellow"
PERM_SESSION = "cyan"

CTX_OK = "green"
CTX_WARN = "yellow"
CTX_CRITICAL = "red"

# =============================================================================
# Branded strings
# =============================================================================

PROMPT_ICON = ">"  # Simple greater-than
PROMPT_TEXT = "godspeed"
BRAND_TAGLINE = "Build fast"

# Syntax theme
SYNTAX_THEME = "monokai"

# =============================================================================
# Ultra-clean markers — Text-only, no emoji
# =============================================================================

MARKER_SUCCESS = "ok"  # Instead of ✓
MARKER_ERROR = "x"  # Instead of ✗
MARKER_WARNING = "!"  # Instead of ⚠
MARKER_TOOL = ">"  # Instead of ▸
MARKER_INFO = "i"  # Instead of ●
MARKER_PARALLEL = "||"  # Instead of ⚡
SEPARATOR_DOT = "|"  # Instead of ·

# Structural
DECORATOR = ""
RULE_CHAR = "-"  # Minimal rule
GUTTER = ""
GUTTER_STYLE = MUTED

# =============================================================================
# Markup helpers
# =============================================================================


def styled(text: str, style: str) -> str:
    """Wrap text in Rich markup tags."""
    return f"[{style}]{text}[/{style}]"


def brand(version: str = "") -> str:
    """Return the branded product name with optional version."""
    name = styled("Godspeed", BOLD_PRIMARY)
    if version:
        return f"{name} {styled(f'v{version}', MUTED)}"
    return name


def icon_prompt(state: str = "") -> str:
    """Return the branded prompt string for prompt-toolkit (HTML format).

    State can be: '' (normal), 'plan' (plan mode), 'paused'.
    """
    color = "ansigold" if not state else "ansiyellow"
    icon = PROMPT_ICON
    suffix = ""
    if state == "plan":
        suffix = " [plan]"
        color = "ansicyan"
    elif state == "paused":
        suffix = " [paused]"
        color = "ansiyellow"
    return f"<b><{color}>{icon} {PROMPT_TEXT}{suffix}></{color}></b> "
