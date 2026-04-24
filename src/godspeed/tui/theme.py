"""Godspeed visual identity — Midnight Gold.
 
Single source of truth for all colors, styles, and branded strings.
Import from here instead of hardcoding Rich markup anywhere else.

Design philosophy — Claude Code meets Godspeed:
- Clean: minimal distraction, maximum information density
- Soft: colors that don't strain eyes over long sessions
- Clear: distinct visual hierarchy for rapid scanning
- Blessed: gold accent reminds you someone's got your back

Palette — refined from Claude Code dark mode:
- Slate 900: deep background (like Claude Code's #1a1a1a)
- Slate 800: elevated surface
- Slate 700: subtle borders
- Slate 600: muted text
- Slate 400: secondary text
- Gold: Godspeed's blessing — brand accent
- Emerald: success (softer than green)
- Rose: error (warm, not aggressive)
- Amber: warning
- Sky: info / tool markers
"""

from __future__ import annotations

# =============================================================================
# Core palette — Rich color names (slate-based like Claude Code)
# =============================================================================

# Base colors — dark mode foundation (Claude Code #1a1a1a inspired)
PRIMARY = "cyan"  # Cyan — clean tool markers
SECONDARY = "grey69"  # Slate — panels, structure
SUCCESS = "green"
ERROR = "red"
WARNING = "yellow"
MUTED = "grey50"  # Slate — secondary text
ACCENT = "cyan"  # Cyan — interactive elements

# Godspeed brand colors
BRAND_GOLD = "yellow1"
BRAND_GOLD_BOLD = "bold yellow1"

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

# Panel borders — subtle, like Claude Code
BORDER_BRAND = "yellow"
BORDER_TOOL = "grey69"
BORDER_INFO = "grey69"
BORDER_SUCCESS = "green"
BORDER_ERROR = "red"
BORDER_WARNING = "yellow"
BORDER_ERROR = "light_coral"
BORDER_WARNING = "gold3"

# Table styles
TABLE_HEADER = "bold cyan"
TABLE_BORDER = "grey46"
TABLE_KEY = "grey35"
TABLE_VALUE = "bold"

# Permission colors
PERM_ALLOW = "green"
PERM_DENY = "red"
PERM_ASK = "yellow"
PERM_SESSION = "cyan"

# Context usage thresholds
CTX_OK = "green"
CTX_WARN = "yellow"
CTX_CRITICAL = "red"

# =============================================================================
# Branded strings
# =============================================================================

PROMPT_ICON = "\u2694"  # ⚔ — crossed swords (builder's mark)
PROMPT_TEXT = "godspeed"
BRAND_TAGLINE = "Security-first — Build with blessing"

# Syntax theme
SYNTAX_THEME = "monokai"

# =============================================================================
# Unicode markers — Claude Code inspired
# =============================================================================

MARKER_SUCCESS = "\u2713"  # ✓ check
MARKER_ERROR = "\u2717"  # ✗ cross
MARKER_WARNING = "\u26a0"  # ⚠ warning
MARKER_TOOL = "\u25b8"  # ▸ tool execution
MARKER_INFO = "\u25cf"  # ● info
MARKER_PARALLEL = "\u26a1"  # ⚡
SEPARATOR_DOT = "\u00b7"  # ·

# Structural
DECORATOR = "\u2571"  # ╲
RULE_CHAR = "\u2500"  # ─
GUTTER = "\u2502"  # │
GUTTER_STYLE = MUTED

# =============================================================================
# Markup helpers
# =============================================================================


def styled(text: str, style: str) -> str:
    """Wrap text in Rich markup tags."""
    return f"[{style}]{text}[/{style}]"


def styled_escaped(text: str, style: str) -> str:
    """Like :func:`styled`, but escape *text* for safe Rich markup.

    For values from the filesystem, shell, or model output (tool paths,
    grep hits, model names). Use :func:`styled` for static UI labels.
    """
    from rich.markup import escape

    return f"[{style}]{escape(text)}[/{style}]"


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
