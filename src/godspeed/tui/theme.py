"""Godspeed visual identity — Midnight Gold.

Single source of truth for all colors, styles, and branded strings.
Import from here instead of hardcoding Rich markup anywhere else.

Palette rationale:
- Gold: speed, confidence, blessing ("Godspeed" = journey blessing)
- Steel blue: trust, precision, engineering
- Mint green: clean modern success
- Warm red: urgent but not aggressive errors
- Amber: clear caution without clashing with gold
- Slate: secondary info, timestamps, muted text
"""

from __future__ import annotations

# =============================================================================
# Core palette — Rich color names
# =============================================================================

PRIMARY = "gold1"  # Electric gold — brand color
SECONDARY = "steel_blue"  # Cool trust — panels, structure
SUCCESS = "green3"  # Mint green — success states
ERROR = "indian_red1"  # Warm red — errors
WARNING = "dark_orange"  # Amber — caution
MUTED = "grey58"  # Slate — secondary text
ACCENT = "cornflower_blue"  # Accent for interactive elements

# =============================================================================
# Semantic styles — Rich markup strings
# =============================================================================

# Text styles
BOLD_PRIMARY = f"bold {PRIMARY}"
BOLD_SECONDARY = f"bold {SECONDARY}"
BOLD_SUCCESS = f"bold {SUCCESS}"
BOLD_ERROR = f"bold {ERROR}"
BOLD_WARNING = f"bold {WARNING}"
DIM = "dim"

# Panel borders
BORDER_BRAND = PRIMARY
BORDER_TOOL = SECONDARY
BORDER_SUCCESS = SUCCESS
BORDER_ERROR = ERROR
BORDER_WARNING = WARNING
BORDER_INFO = SECONDARY

# Table styles
TABLE_HEADER = f"bold {PRIMARY}"
TABLE_BORDER = SECONDARY
TABLE_KEY = MUTED
TABLE_VALUE = "bold"

# Permission colors
PERM_ALLOW = SUCCESS
PERM_DENY = ERROR
PERM_ASK = WARNING
PERM_SESSION = ACCENT

# Context usage thresholds
CTX_OK = SUCCESS
CTX_WARN = WARNING
CTX_CRITICAL = ERROR

# =============================================================================
# Branded strings
# =============================================================================

PROMPT_ICON = "\u26a1"  # Lightning bolt
PROMPT_TEXT = "godspeed"
BRAND_TAGLINE = "Security-first coding agent"

# Syntax theme for all code blocks
SYNTAX_THEME = "monokai"

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
