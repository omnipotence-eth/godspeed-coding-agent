"""Tests for the repo map tool and tree-sitter parser."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest

from godspeed.context.repo_map import RepoMapper, Symbol
from godspeed.tools.base import ToolContext
from godspeed.tools.repo_map import RepoMapTool

_has_tree_sitter = importlib.util.find_spec("tree_sitter_language_pack") is not None


@pytest.fixture
def mapper() -> RepoMapper:
    return RepoMapper()


@pytest.fixture
def tool() -> RepoMapTool:
    return RepoMapTool()


_skip_no_treesitter = pytest.mark.skipif(
    not _has_tree_sitter,
    reason="requires godspeed[context]",
)


@_skip_no_treesitter
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

    def test_parse_decorated_class(self, mapper: RepoMapper, tmp_path: Path) -> None:
        f = tmp_path / "dc.py"
        f.write_text("@dataclass\nclass Foo:\n    x: int\n    def bar(self):\n        pass\n")
        symbols = mapper.parse_file(f)
        assert len(symbols) == 1
        assert symbols[0].kind == "class"

    def test_parse_decorated_returns_none_for_unknown(
        self, mapper: RepoMapper, tmp_path: Path
    ) -> None:
        f = tmp_path / "odd.py"
        f.write_text("@something\nx = 1\n")
        symbols = mapper.parse_file(f)
        # No function/class under decorator, so no symbols
        assert isinstance(symbols, list)

    def test_parse_empty_file(self, mapper: RepoMapper, tmp_path: Path) -> None:
        f = tmp_path / "empty.py"
        f.write_text("")
        symbols = mapper.parse_file(f)
        assert symbols == []

    def test_parse_binary_file_graceful(self, mapper: RepoMapper, tmp_path: Path) -> None:
        f = tmp_path / "data.py"
        f.write_bytes(b"\x00\x01\x02\x03")
        symbols = mapper.parse_file(f)
        assert isinstance(symbols, list)

    def test_parse_unsupported_extension(self, mapper: RepoMapper, tmp_path: Path) -> None:
        f = tmp_path / "data.csv"
        f.write_text("a,b,c\n1,2,3\n")
        symbols = mapper.parse_file(f)
        assert symbols == []

    def test_parse_no_read_permission(self, mapper: RepoMapper, tmp_path: Path) -> None:
        f = tmp_path / "locked.py"
        f.write_text("def foo(): pass\n")
        with patch.object(Path, "read_bytes", side_effect=OSError("Permission denied")):
            symbols = mapper.parse_file(f)
            assert symbols == []

    def test_parse_oserror_read(self, mapper: RepoMapper, tmp_path: Path) -> None:
        f = tmp_path / "err.py"
        f.write_text("def foo(): pass\n")
        with patch.object(Path, "read_bytes", side_effect=PermissionError("no read")):
            symbols = mapper.parse_file(f)
            assert symbols == []

    def test_parse_python_syntax_error(self, mapper: RepoMapper, tmp_path: Path) -> None:
        f = tmp_path / "bad.py"
        f.write_text("def broken(\n")
        symbols = mapper.parse_file(f)
        assert isinstance(symbols, list)

    def test_parse_javascript(self, mapper: RepoMapper, tmp_path: Path) -> None:
        f = tmp_path / "app.js"
        f.write_text("function greet(name) {\n  return 'hello ' + name;\n}\n")
        symbols = mapper.parse_file(f)
        assert len(symbols) == 1
        assert symbols[0].name == "greet"

    def test_parse_javascript_class(self, mapper: RepoMapper, tmp_path: Path) -> None:
        f = tmp_path / "app.js"
        f.write_text("class Animal {\n  speak() {\n    return 'hi';\n  }\n}\n")
        symbols = mapper.parse_file(f)
        assert len(symbols) == 1
        assert symbols[0].kind == "class"

    def test_parse_javascript_export(self, mapper: RepoMapper, tmp_path: Path) -> None:
        f = tmp_path / "app.js"
        f.write_text("export function helper() {\n  return 1;\n}\n")
        symbols = mapper.parse_file(f)
        assert len(symbols) == 1
        assert symbols[0].name == "helper"

    def test_parse_typescript(self, mapper: RepoMapper, tmp_path: Path) -> None:
        f = tmp_path / "app.ts"
        f.write_text("function add(a: number, b: number): number {\n  return a + b;\n}\n")
        symbols = mapper.parse_file(f)
        assert len(symbols) >= 1

    def test_parse_go_file(self, mapper: RepoMapper, tmp_path: Path) -> None:
        f = tmp_path / "main.go"
        f.write_text(
            'package main\n\nfunc main() {\n\tprintln("hello")\n}\n\n'
            "func (s *Server) Start() error {\n\treturn nil\n}\n"
        )
        symbols = mapper.parse_file(f)
        assert len(symbols) >= 1

    def test_parse_go_type_declaration(self, mapper: RepoMapper, tmp_path: Path) -> None:
        f = tmp_path / "types.go"
        f.write_text("package main\n\ntype MyInt int\n\ntype MyStruct struct {\n\tX int\n}\n")
        symbols = mapper.parse_file(f)
        assert isinstance(symbols, list)

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

    def test_map_directory_not_a_directory(self, mapper: RepoMapper, tmp_path: Path) -> None:
        f = tmp_path / "file.py"
        f.write_text("x = 1\n")
        result = mapper.map_directory(f)
        assert "Not a directory" in result

    def test_map_directory_no_recognized_files(self, mapper: RepoMapper, tmp_path: Path) -> None:
        (tmp_path / "data.csv").write_text("a,b\n1,2\n")
        (tmp_path / "readme.txt").write_text("hello\n")
        result = mapper.map_directory(tmp_path)
        assert "No symbols found" in result

    def test_map_directory_file_with_no_symbols_skipped(
        self, mapper: RepoMapper, tmp_path: Path
    ) -> None:
        (tmp_path / "empty.py").write_text("")
        (tmp_path / "actual.py").write_text("def foo(): pass\n")
        result = mapper.map_directory(tmp_path)
        assert "foo" in result
        assert "empty" not in result

    def test_parse_file_with_parse_exception(self, mapper: RepoMapper, tmp_path: Path) -> None:
        f = tmp_path / "bad_parse.py"
        f.write_text("def foo():\n    pass\n")
        with patch.object(mapper, "_get_parser", side_effect=Exception("parse crash")):
            symbols = mapper.parse_file(f)
            assert symbols == []

    def test_parse_decorated_unknown_returns_none(self, mapper: RepoMapper, tmp_path: Path) -> None:
        f = tmp_path / "dec.py"
        f.write_text("@decorator\n@another\ndef foo(): pass\n")
        symbols = mapper.parse_file(f)
        assert len(symbols) >= 1

    def test_parse_export_nothing_inside(self, mapper: RepoMapper, tmp_path: Path) -> None:
        f = tmp_path / "exp.js"
        f.write_text("export { x };\nfunction bar() {}\n")
        symbols = mapper.parse_file(f)
        assert any(s.name == "bar" for s in symbols)

    def test_symbol_format_compact(self) -> None:
        s = Symbol(name="single", kind="function", line=42)
        assert s.format() == "single(L42)"


class TestRepoMapperUnavailable:
    """Test RepoMapper when tree-sitter is NOT available."""

    def test_available_false_when_not_installed(self) -> None:
        with patch.object(RepoMapper, "_check_availability", return_value=False):
            mapper = RepoMapper()
            assert mapper.available is False

    def test_parse_file_returns_empty_when_unavailable(self) -> None:
        with patch.object(RepoMapper, "_check_availability", return_value=False):
            mapper = RepoMapper()
            result = mapper.parse_file(Path("test.py"))
            assert result == []

    def test_map_directory_returns_message_when_unavailable(self) -> None:
        with patch.object(RepoMapper, "_check_availability", return_value=False):
            mapper = RepoMapper()
            result = mapper.map_directory(Path("/tmp"))
            assert "not available" in result

    def test_import_error_during_check_availability(self) -> None:
        with patch.dict("sys.modules", {"tree_sitter_language_pack": None}):
            mapper = RepoMapper()
            assert mapper.available is False


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

    def test_format_nested_children(self) -> None:
        parent = Symbol(name="A", kind="class", line=1)
        child = Symbol(name="B", kind="class", line=2)
        child.children.append(Symbol(name="foo", kind="method", line=3))
        parent.children.append(child)
        formatted = parent.format()
        assert "A(L1)" in formatted
        assert "  B(L2)" in formatted
        assert "    foo(L3)" in formatted


@_skip_no_treesitter
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
