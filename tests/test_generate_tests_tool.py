"""Tests for the generate_tests tool (v2.8.0)."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from godspeed.tools.base import ToolContext
from godspeed.tools.generate_tests import (
    MAX_SOURCE_CHARS,
    GenerateTestsTool,
    _clean_llm_output,
    _display_rel,
    _module_name_from,
    _resolve_output_path,
)


class _FakeLLMResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeLLM:
    def __init__(self, content: str) -> None:
        self._content = content
        self.chat = AsyncMock(return_value=_FakeLLMResponse(content))


class TestModuleNameFrom:
    def test_src_layout_strips_src_prefix(self, tmp_path: Path) -> None:
        f = tmp_path / "src" / "pkg" / "mod.py"
        f.parent.mkdir(parents=True)
        f.touch()
        assert _module_name_from(f, tmp_path) == "pkg.mod"

    def test_flat_layout(self, tmp_path: Path) -> None:
        f = tmp_path / "util.py"
        f.touch()
        assert _module_name_from(f, tmp_path) == "util"

    def test_nested_without_src(self, tmp_path: Path) -> None:
        f = tmp_path / "lib" / "nested" / "thing.py"
        f.parent.mkdir(parents=True)
        f.touch()
        assert _module_name_from(f, tmp_path) == "lib.nested.thing"

    def test_outside_cwd_falls_back_to_stem(self, tmp_path: Path) -> None:
        # Path that can't be made relative → stem fallback.
        f = Path("/elsewhere/foo.py")
        assert _module_name_from(f, tmp_path) == "foo"


class TestCleanLlmOutput:
    def test_strips_triple_backtick_fence(self) -> None:
        raw = "```python\ndef test_x():\n    assert 1\n```"
        assert _clean_llm_output(raw).strip() == "def test_x():\n    assert 1"

    def test_strips_plain_fence(self) -> None:
        raw = "```\nimport pytest\n```"
        assert _clean_llm_output(raw).strip() == "import pytest"

    def test_passes_unfenced_through(self) -> None:
        raw = "def test_x():\n    assert 1\n"
        assert _clean_llm_output(raw) == raw

    def test_adds_trailing_newline(self) -> None:
        raw = "def test_x():\n    assert 1"
        assert _clean_llm_output(raw).endswith("\n")


class TestGenerateTestsTool:
    @pytest.mark.asyncio
    async def test_missing_llm_client_returns_clear_error(self, tmp_path: Path) -> None:
        src = tmp_path / "thing.py"
        src.write_text("def f(): return 1", encoding="utf-8")
        ctx = ToolContext(cwd=tmp_path, session_id="t", llm_client=None)
        tool = GenerateTestsTool()
        result = await tool.execute({"source_path": "thing.py"}, ctx)
        assert result.is_error
        assert "llm_client" in (result.error or "")

    @pytest.mark.asyncio
    async def test_source_path_required(self, tmp_path: Path) -> None:
        ctx = ToolContext(cwd=tmp_path, session_id="t", llm_client=_FakeLLM("x"))
        tool = GenerateTestsTool()
        result = await tool.execute({}, ctx)
        assert result.is_error
        assert "source_path" in (result.error or "")

    @pytest.mark.asyncio
    async def test_missing_source_file_reports_clearly(self, tmp_path: Path) -> None:
        ctx = ToolContext(cwd=tmp_path, session_id="t", llm_client=_FakeLLM("x"))
        tool = GenerateTestsTool()
        result = await tool.execute({"source_path": "nope.py"}, ctx)
        assert result.is_error
        assert "does not exist" in (result.error or "")

    @pytest.mark.asyncio
    async def test_happy_path_writes_test_file(self, tmp_path: Path) -> None:
        src = tmp_path / "util.py"
        src.write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
        llm = _FakeLLM(
            "```python\n"
            "from util import add\n\n"
            "def test_add() -> None:\n"
            "    assert add(1, 2) == 3\n"
            "```"
        )
        ctx = ToolContext(cwd=tmp_path, session_id="t", llm_client=llm)
        tool = GenerateTestsTool()

        result = await tool.execute({"source_path": "util.py"}, ctx)

        assert not result.is_error
        out = tmp_path / "tests" / "test_util.py"
        assert out.is_file()
        content = out.read_text(encoding="utf-8")
        assert "from util import add" in content
        assert "assert add(1, 2) == 3" in content
        # Markdown fences were stripped.
        assert "```" not in content

    @pytest.mark.asyncio
    async def test_custom_output_path_honored(self, tmp_path: Path) -> None:
        src = tmp_path / "util.py"
        src.write_text("x = 1\n", encoding="utf-8")
        llm = _FakeLLM("def test_x():\n    assert 1\n")
        ctx = ToolContext(cwd=tmp_path, session_id="t", llm_client=llm)
        tool = GenerateTestsTool()

        result = await tool.execute(
            {"source_path": "util.py", "output_path": "custom/here.py"}, ctx
        )

        assert not result.is_error
        assert (tmp_path / "custom" / "here.py").is_file()

    @pytest.mark.asyncio
    async def test_empty_llm_response_errors(self, tmp_path: Path) -> None:
        src = tmp_path / "util.py"
        src.write_text("x = 1", encoding="utf-8")
        ctx = ToolContext(cwd=tmp_path, session_id="t", llm_client=_FakeLLM("   "))
        tool = GenerateTestsTool()
        result = await tool.execute({"source_path": "util.py"}, ctx)
        assert result.is_error
        assert "empty" in (result.error or "").lower()

    @pytest.mark.asyncio
    async def test_llm_exception_reported(self, tmp_path: Path) -> None:
        src = tmp_path / "util.py"
        src.write_text("x = 1", encoding="utf-8")

        class _BrokenLLM:
            async def chat(self, messages):
                raise RuntimeError("provider 500")

        ctx = ToolContext(cwd=tmp_path, session_id="t", llm_client=_BrokenLLM())
        tool = GenerateTestsTool()
        result = await tool.execute({"source_path": "util.py"}, ctx)
        assert result.is_error
        assert "LLM call failed" in (result.error or "")

    @pytest.mark.asyncio
    async def test_source_too_large(self, tmp_path: Path) -> None:
        src = tmp_path / "big.py"
        big_content = "x = 1\n" * (MAX_SOURCE_CHARS // 3)
        src.write_text(big_content, encoding="utf-8")
        ctx = ToolContext(cwd=tmp_path, session_id="t", llm_client=_FakeLLM("x"))
        tool = GenerateTestsTool()
        result = await tool.execute({"source_path": "big.py"}, ctx)
        assert result.is_error
        assert "too large" in (result.error or "").lower() or "max is" in (result.error or "").lower()

    @pytest.mark.asyncio
    async def test_resolve_tool_path_value_error(self, tmp_path: Path) -> None:
        src = tmp_path / "util.py"
        src.write_text("x = 1", encoding="utf-8")
        ctx = ToolContext(cwd=tmp_path, session_id="t", llm_client=_FakeLLM("x"))
        tool = GenerateTestsTool()
        with patch(
            "godspeed.tools.generate_tests.resolve_tool_path",
            side_effect=ValueError("Access denied"),
        ):
            result = await tool.execute({"source_path": "util.py"}, ctx)
        assert result.is_error
        assert "Access denied" in result.error

    @pytest.mark.asyncio
    async def test_oserror_reading_source(self, tmp_path: Path) -> None:
        src = tmp_path / "util.py"
        src.write_text("x = 1", encoding="utf-8")
        ctx = ToolContext(cwd=tmp_path, session_id="t", llm_client=_FakeLLM("x"))
        tool = GenerateTestsTool()
        with patch("pathlib.Path.read_text", side_effect=OSError("permission denied")):
            result = await tool.execute({"source_path": "util.py"}, ctx)
        assert result.is_error
        assert "Could not read" in result.error

    @pytest.mark.asyncio
    async def test_oserror_writing_test_file(self, tmp_path: Path) -> None:
        src = tmp_path / "util.py"
        src.write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
        llm = _FakeLLM("def test_add():\n    assert 1\n")
        ctx = ToolContext(cwd=tmp_path, session_id="t", llm_client=llm)
        tool = GenerateTestsTool()
        with patch("pathlib.Path.write_text", side_effect=OSError("disk full")):
            result = await tool.execute({"source_path": "util.py"}, ctx)
        assert result.is_error
        assert "Could not write" in result.error

    @pytest.mark.asyncio
    async def test_absolute_output_path(self, tmp_path: Path) -> None:
        src = tmp_path / "util.py"
        src.write_text("x = 1\n", encoding="utf-8")
        llm = _FakeLLM("def test_x():\n    assert 1\n")
        ctx = ToolContext(cwd=tmp_path, session_id="t", llm_client=llm)
        tool = GenerateTestsTool()

        abs_path = tmp_path / "abs_tests" / "test_abs.py"
        result = await tool.execute(
            {"source_path": "util.py", "output_path": str(abs_path)}, ctx
        )
        assert not result.is_error
        assert abs_path.is_file()


class TestResolveOutputPath:
    def test_absolute_path(self, tmp_path: Path) -> None:
        abs_path = tmp_path / "out" / "test_x.py"
        result = _resolve_output_path(str(abs_path), tmp_path / "src" / "util.py", tmp_path)
        assert result == abs_path

    def test_relative_path(self, tmp_path: Path) -> None:
        result = _resolve_output_path("custom/test_util.py", tmp_path / "src" / "util.py", tmp_path)
        assert result == tmp_path / "custom" / "test_util.py"

    def test_default_path(self, tmp_path: Path) -> None:
        result = _resolve_output_path(None, tmp_path / "src" / "util.py", tmp_path)
        assert result == tmp_path / "tests" / "test_util.py"

    def test_empty_string_defaults(self, tmp_path: Path) -> None:
        result = _resolve_output_path("  ", tmp_path / "src" / "util.py", tmp_path)
        assert result == tmp_path / "tests" / "test_util.py"


class TestCleanLlmOutputExtra:
    def test_closing_fence_line_exact(self) -> None:
        raw = "```\ndef test_x():\n    pass\n```"
        cleaned = _clean_llm_output(raw)
        assert "```" not in cleaned

    def test_no_closing_fence(self) -> None:
        raw = "```python\ndef test_x():\n    pass"
        cleaned = _clean_llm_output(raw)
        assert cleaned.strip() == "def test_x():\n    pass"

    def test_only_fence_no_content(self) -> None:
        raw = "```\n```"
        cleaned = _clean_llm_output(raw)
        assert cleaned.strip() == ""


class TestDisplayRel:
    def test_within_cwd(self, tmp_path: Path) -> None:
        f = tmp_path / "tests" / "test_x.py"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.touch()
        assert _display_rel(f, tmp_path) == "tests\\test_x.py" if os.name == "nt" else "tests/test_x.py"

    def test_outside_cwd(self, tmp_path: Path) -> None:
        f = Path("/absolute/path/test_x.py")
        assert _display_rel(f, tmp_path) == str(f)
