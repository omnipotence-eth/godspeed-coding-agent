"""Tests for the Midnight Gold theme module."""

from __future__ import annotations

from godspeed.tui.theme import (
    ACCENT,
    BOLD_ERROR,
    BOLD_PRIMARY,
    BOLD_SECONDARY,
    BOLD_SUCCESS,
    BOLD_WARNING,
    BORDER_BRAND,
    BORDER_ERROR,
    BORDER_INFO,
    BORDER_SUCCESS,
    BORDER_TOOL,
    BORDER_WARNING,
    BRAND_TAGLINE,
    CTX_CRITICAL,
    CTX_OK,
    CTX_WARN,
    DIM,
    ERROR,
    MUTED,
    PERM_ALLOW,
    PERM_ASK,
    PERM_DENY,
    PERM_SESSION,
    PRIMARY,
    PROMPT_ICON,
    PROMPT_TEXT,
    SECONDARY,
    SUCCESS,
    SYNTAX_THEME,
    TABLE_BORDER,
    TABLE_HEADER,
    TABLE_KEY,
    TABLE_VALUE,
    WARNING,
    brand,
    icon_prompt,
    styled,
)


class TestPaletteConstants:
    """Verify palette constants are non-empty strings."""

    def test_primary_colors(self) -> None:
        for color in [PRIMARY, SECONDARY, SUCCESS, ERROR, WARNING, MUTED, ACCENT]:
            assert isinstance(color, str)
            assert len(color) > 0

    def test_bold_styles_contain_bold(self) -> None:
        for style in [BOLD_PRIMARY, BOLD_SECONDARY, BOLD_SUCCESS, BOLD_ERROR, BOLD_WARNING]:
            assert "bold" in style

    def test_border_styles_are_strings(self) -> None:
        for border in [
            BORDER_BRAND,
            BORDER_TOOL,
            BORDER_SUCCESS,
            BORDER_ERROR,
            BORDER_WARNING,
            BORDER_INFO,
        ]:
            assert isinstance(border, str)
            assert len(border) > 0

    def test_table_styles(self) -> None:
        assert isinstance(TABLE_HEADER, str)
        assert isinstance(TABLE_BORDER, str)
        assert isinstance(TABLE_KEY, str)
        assert isinstance(TABLE_VALUE, str)

    def test_permission_colors(self) -> None:
        for color in [PERM_ALLOW, PERM_DENY, PERM_ASK, PERM_SESSION]:
            assert isinstance(color, str)
            assert len(color) > 0

    def test_context_threshold_colors(self) -> None:
        for color in [CTX_OK, CTX_WARN, CTX_CRITICAL]:
            assert isinstance(color, str)
            assert len(color) > 0

    def test_dim_style(self) -> None:
        assert DIM == "dim"

    def test_syntax_theme(self) -> None:
        assert isinstance(SYNTAX_THEME, str)
        assert len(SYNTAX_THEME) > 0


class TestMarkerConstants:
    """Test Unicode marker constants."""

    def test_marker_success(self) -> None:
        from godspeed.tui.theme import MARKER_SUCCESS

        assert MARKER_SUCCESS == "\u2713"

    def test_marker_error(self) -> None:
        from godspeed.tui.theme import MARKER_ERROR

        assert MARKER_ERROR == "\u2717"

    def test_marker_warning(self) -> None:
        from godspeed.tui.theme import MARKER_WARNING

        assert MARKER_WARNING == "\u26a0"

    def test_marker_tool(self) -> None:
        from godspeed.tui.theme import MARKER_TOOL

        assert MARKER_TOOL == "\u25b8"

    def test_separator_dot(self) -> None:
        from godspeed.tui.theme import SEPARATOR_DOT

        assert SEPARATOR_DOT == "\u00b7"

    def test_marker_info(self) -> None:
        from godspeed.tui.theme import MARKER_INFO

        assert MARKER_INFO == "\u25cf"


class TestStructuralConstants:
    """Test Crush-inspired structural character constants."""

    def test_decorator(self) -> None:
        from godspeed.tui.theme import DECORATOR

        assert DECORATOR == "\u2571"

    def test_rule_char(self) -> None:
        from godspeed.tui.theme import RULE_CHAR

        assert RULE_CHAR == "\u2500"

    def test_gutter(self) -> None:
        from godspeed.tui.theme import GUTTER

        assert GUTTER == "\u2502"

    def test_gutter_style_is_muted(self) -> None:
        from godspeed.tui.theme import GUTTER_STYLE, MUTED

        assert GUTTER_STYLE == MUTED


class TestBrandedStrings:
    """Test branded string constants."""

    def test_prompt_icon_is_lightning(self) -> None:
        assert PROMPT_ICON == "\u2694"  # ⚔ crossed swords (builder's mark)

    def test_prompt_text(self) -> None:
        assert PROMPT_TEXT == "godspeed"

    def test_brand_tagline(self) -> None:
        assert "security" in BRAND_TAGLINE.lower() or "coding" in BRAND_TAGLINE.lower()


class TestStyled:
    """Test the styled() markup helper."""

    def test_wraps_text_in_markup(self) -> None:
        result = styled("hello", "bold red")
        assert result == "[bold red]hello[/bold red]"

    def test_empty_text(self) -> None:
        result = styled("", "dim")
        assert result == "[dim][/dim]"

    def test_nested_style(self) -> None:
        result = styled("test", BOLD_PRIMARY)
        assert f"[{BOLD_PRIMARY}]test[/{BOLD_PRIMARY}]" == result


class TestBrand:
    """Test the brand() helper."""

    def test_brand_without_version(self) -> None:
        result = brand()
        assert "Godspeed" in result
        assert BOLD_PRIMARY in result

    def test_brand_with_version(self) -> None:
        result = brand("1.2.3")
        assert "Godspeed" in result
        assert "v1.2.3" in result
        assert MUTED in result

    def test_brand_empty_version(self) -> None:
        result = brand("")
        # Empty string is falsy, so no version shown
        assert "v" not in result


class TestIconPrompt:
    """Test the icon_prompt() helper for prompt-toolkit HTML."""

    def test_default_prompt(self) -> None:
        result = icon_prompt()
        assert PROMPT_ICON in result
        assert PROMPT_TEXT in result
        assert "ansigold" in result

    def test_plan_mode_prompt(self) -> None:
        result = icon_prompt("plan")
        assert "[plan]" in result
        assert "ansicyan" in result

    def test_paused_prompt(self) -> None:
        result = icon_prompt("paused")
        assert "[paused]" in result
        assert "ansiyellow" in result

    def test_unknown_state_uses_yellow(self) -> None:
        result = icon_prompt("something")
        assert "ansiyellow" in result

    def test_prompt_is_html_formatted(self) -> None:
        result = icon_prompt()
        assert "<b>" in result
        assert "</b>" in result


class TestRichCompatibility:
    """Verify theme constants work with Rich rendering."""

    def test_styled_renders_without_error(self) -> None:
        from io import StringIO

        from rich.console import Console
        from rich.text import Text

        buf = StringIO()
        c = Console(file=buf, force_terminal=True, width=80)
        text = Text.from_markup(styled("test", BOLD_PRIMARY))
        c.print(text)
        assert "test" in buf.getvalue()

    def test_all_border_styles_valid(self) -> None:
        from io import StringIO

        from rich.console import Console
        from rich.panel import Panel

        for border in [
            BORDER_BRAND,
            BORDER_TOOL,
            BORDER_SUCCESS,
            BORDER_ERROR,
            BORDER_WARNING,
            BORDER_INFO,
        ]:
            buf = StringIO()
            c = Console(file=buf, force_terminal=True, width=80)
            panel = Panel("content", border_style=border)
            c.print(panel)
            assert "content" in buf.getvalue()
