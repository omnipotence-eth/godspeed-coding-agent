"""Tests for the repo map tool and tree-sitter parser."""

from __future__ import annotations

from pathlib import Path

import pytest

from godspeed.context.repo_map import RepoMapper, Symbol
from godspeed.tools.base import ToolContext
from godspeed.tools.repo_map import RepoMapTool


@pytest.fixture
def mapper() -> RepoMapper:
    return RepoMapper()


@pytest.fixture
def tool() -> RepoMapTool:
    return RepoMapTool()


class TestRepoMapper:
    """Test the tree-sitter symbol extraction."""

    def test_available(self, mapper: RepoMapper) -> None:
        assert mapper.available is True

    def test_parse_python_functions(self, mapper: RepoMapper, tmp_path: Path) -> None:
        f = tmp_path / "module.py"
        f.write_text("def foo():\n    pass\n\ndef bar(x: int) -> str:\n    return str(x)\n")
        symbols = mapper.parse_file(f)
        assert len(symbols) == 2
        assert symbols[0].name == "foo"
        assert symbols[0].kind == "function"
        assert symbols[0].line == 1
        assert symbols[1].name == "bar"
        assert symbols[1].line == 4

    def test_parse_python_class_with_methods(self, mapper: RepoMapper, tmp_path: Path) -> None:
        f = tmp_path / "models.py"
        f.write_text(
            "class User:\n"
            "    def __init__(self):\n"
            "        pass\n"
            "    def save(self):\n"
            "        pass\n"
        )
        symbols = mapper.parse_file(f)
        assert len(symbols) == 1
        assert symbols[0].name == "User"
        assert symbols[0].kind == "class"
        assert len(symbols[0].children) == 2
        assert symbols[0].children[0].name == "__init__"
        assert symbols[0].children[1].name == "save"

    def test_parse_decorated_function(self, mapper: RepoMapper, tmp_path: Path) -> None:
        f = tmp_path / "app.py"
        f.write_text("@property\ndef name(self):\n    return self._name\n")
        symbols = mapper.parse_file(f)
        assert len(symbols) == 1
        assert symbols[0].name == "name"

    def test_parse_empty_file(self, mapper: RepoMapper, tmp_path: Path) -> None:
        f = tmp_path / "empty.py"
        f.write_text("")
        symbols = mapper.parse_file(f)
        assert symbols == []

    def test_parse_binary_file_graceful(self, mapper: RepoMapper, tmp_path: Path) -> None:
        f = tmp_path / "data.py"
        f.write_bytes(b"\x00\x01\x02\x03")
        # Should not crash — may return empty or partial
        symbols = mapper.parse_file(f)
        assert isinstance(symbols, list)

    def test_parse_unsupported_extension(self, mapper: RepoMapper, tmp_path: Path) -> None:
        f = tmp_path / "data.csv"
        f.write_text("a,b,c\n1,2,3\n")
        symbols = mapper.parse_file(f)
        assert symbols == []

    def test_parse_javascript(self, mapper: RepoMapper, tmp_path: Path) -> None:
        f = tmp_path / "app.js"
        f.write_text("function greet(name) {\n  return 'hello ' + name;\n}\n")
        symbols = mapper.parse_file(f)
        assert len(symbols) == 1
        assert symbols[0].name == "greet"

    def test_map_directory(self, mapper: RepoMapper, tmp_path: Path) -> None:
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("def main():\n    pass\n")
        (tmp_path / "src" / "utils.py").write_text(
            "class Helper:\n    def run(self):\n        pass\n"
        )
        result = mapper.map_directory(tmp_path)
        assert "main" in result
        assert "Helper" in result
        assert "run" in result

    def test_map_directory_respects_excludes(self, mapper: RepoMapper, tmp_path: Path) -> None:
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "app.py").write_text("def app(): pass\n")
        (tmp_path / "node_modules").mkdir()
        (tmp_path / "node_modules" / "junk.py").write_text("def junk(): pass\n")
        result = mapper.map_directory(tmp_path)
        assert "app" in result
        assert "junk" not in result

    def test_map_directory_pattern_filter(self, mapper: RepoMapper, tmp_path: Path) -> None:
        (tmp_path / "main.py").write_text("def main(): pass\n")
        (tmp_path / "app.js").write_text("function app() {}\n")
        result = mapper.map_directory(tmp_path, pattern="*.py")
        assert "main" in result
        assert "app" not in result

    def test_map_empty_directory(self, mapper: RepoMapper, tmp_path: Path) -> None:
        result = mapper.map_directory(tmp_path)
        assert "No symbols found" in result

    def test_map_directory_max_depth(self, mapper: RepoMapper, tmp_path: Path) -> None:
        deep = tmp_path / "a" / "b" / "c" / "d" / "e" / "f"
        deep.mkdir(parents=True)
        (deep / "deep.py").write_text("def deep(): pass\n")
        (tmp_path / "shallow.py").write_text("def shallow(): pass\n")
        result = mapper.map_directory(tmp_path, max_depth=2)
        assert "shallow" in result
        assert "deep" not in result


class TestSymbol:
    """Test Symbol formatting."""

    def test_format_simple(self) -> None:
        s = Symbol(name="foo", kind="function", line=10)
        assert s.format() == "foo(L10)"

    def test_format_with_children(self) -> None:
        parent = Symbol(name="MyClass", kind="class", line=1)
        parent.children.append(Symbol(name="method", kind="method", line=3))
        formatted = parent.format()
        assert "MyClass(L1)" in formatted
        assert "  method(L3)" in formatted


class TestRepoMapTool:
    """Test the tool wrapper."""

    def test_metadata(self, tool: RepoMapTool) -> None:
        assert tool.name == "repo_map"
        assert tool.risk_level == "read_only"

    @pytest.mark.asyncio
    async def test_map_project_root(self, tool: RepoMapTool, tool_context: ToolContext) -> None:
        (tool_context.cwd / "main.py").write_text("def main():\n    pass\n")
        result = await tool.execute({}, tool_context)
        assert not result.is_error
        assert "main" in result.output

    @pytest.mark.asyncio
    async def test_map_subdirectory(self, tool: RepoMapTool, tool_context: ToolContext) -> None:
        src = tool_context.cwd / "src"
        src.mkdir()
        (src / "app.py").write_text("class App:\n    def run(self): pass\n")
        result = await tool.execute({"path": "src"}, tool_context)
        assert not result.is_error
        assert "App" in result.output

    @pytest.mark.asyncio
    async def test_not_a_directory(self, tool: RepoMapTool, tool_context: ToolContext) -> None:
        f = tool_context.cwd / "file.py"
        f.write_text("x = 1\n")
        result = await tool.execute({"path": "file.py"}, tool_context)
        assert result.is_error
        assert "not a directory" in result.error.lower()

    @pytest.mark.asyncio
    async def test_with_pattern(self, tool: RepoMapTool, tool_context: ToolContext) -> None:
        (tool_context.cwd / "a.py").write_text("def a(): pass\n")
        (tool_context.cwd / "b.js").write_text("function b() {}\n")
        result = await tool.execute({"pattern": "*.py"}, tool_context)
        assert not result.is_error
        assert "a" in result.output
