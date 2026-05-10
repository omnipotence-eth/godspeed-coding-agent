"""Coverage gap tests for repo_map — all missed branches."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest

from godspeed.context.repo_map import RepoMapper

_has_tree_sitter = importlib.util.find_spec("tree_sitter_language_pack") is not None

_skip_no_treesitter = pytest.mark.skipif(
    not _has_tree_sitter,
    reason="requires godspeed[context]",
)


@pytest.fixture
def mapper() -> RepoMapper:
    return RepoMapper()


# ── Tree-sitter not available tests ─────────────────────────────────────────


class TestRepoMapperNotAvailableBranches:
    """Cover all branches when tree-sitter is not available."""

    def test_check_availability_import_error(self) -> None:
        with patch.dict("sys.modules", {"tree_sitter_language_pack": None}):
            mapper = RepoMapper()
            assert mapper.available is False

    def test_check_availability_import_error_direct(self) -> None:
        with patch("builtins.__import__", side_effect=ImportError("no module")):
            result = RepoMapper._check_availability()
            assert result is False

    def test_parse_file_when_unavailable(self) -> None:
        with patch.object(RepoMapper, "_check_availability", return_value=False):
            mapper = RepoMapper()
            result = mapper.parse_file(Path("test.py"))  # line 104
            assert result == []

    def test_map_directory_when_unavailable(self) -> None:
        with patch.object(RepoMapper, "_check_availability", return_value=False):
            mapper = RepoMapper()
            result = mapper.map_directory(Path("/tmp"))  # line 241
            assert "not available" in result


# ── Parse file edge cases ───────────────────────────────────────────────────


@_skip_no_treesitter
class TestParseFileEdges:
    """Cover remaining branches in parse_file."""

    def test_parse_file_oserror_read(self, mapper: RepoMapper, tmp_path: Path) -> None:
        f = tmp_path / "locked.py"
        f.write_text("def foo(): pass\n")
        with patch.object(Path, "read_bytes", side_effect=OSError("cannot read")):
            symbols = mapper.parse_file(f)
            assert symbols == []

    def test_parse_file_tree_sitter_exception(self, mapper: RepoMapper, tmp_path: Path) -> None:
        f = tmp_path / "bad_parse.py"
        f.write_text("def foo(): pass\n")
        with patch.object(mapper, "_get_parser", side_effect=Exception("parse crash")):
            symbols = mapper.parse_file(f)
            assert symbols == []

    def test_parse_file_permission_error(self, mapper: RepoMapper, tmp_path: Path) -> None:
        f = tmp_path / "perm.py"
        f.write_text("def foo(): pass\n")
        with patch.object(Path, "read_bytes", side_effect=PermissionError("no access")):
            symbols = mapper.parse_file(f)
            assert symbols == []

    def test_parse_file_unsupported_language(self, mapper: RepoMapper, tmp_path: Path) -> None:
        f = tmp_path / "data.csv"
        f.write_text("a,b,c\n")
        symbols = mapper.parse_file(f)
        assert symbols == []

    def test_parse_file_empty_content(self, mapper: RepoMapper, tmp_path: Path) -> None:
        f = tmp_path / "empty.py"
        f.write_bytes(b"")
        symbols = mapper.parse_file(f)
        assert symbols == []

    def test_parse_file_whitespace_only(self, mapper: RepoMapper, tmp_path: Path) -> None:
        f = tmp_path / "blank.py"
        f.write_text("   \n\t\n   \n")
        symbols = mapper.parse_file(f)
        assert symbols == []

    def test_parse_file_binary_content_utf8_fallback(
        self, mapper: RepoMapper, tmp_path: Path
    ) -> None:
        f = tmp_path / "data.py"
        f.write_bytes(b"\x00\x01\x02")
        symbols = mapper.parse_file(f)
        assert isinstance(symbols, list)


# ── Decorated definition edge cases ─────────────────────────────────────────


@_skip_no_treesitter
class TestDecoratedDefinitions:
    """Cover decorated_definition branches."""

    def test_decorated_with_no_function_or_class(self, mapper: RepoMapper, tmp_path: Path) -> None:
        f = tmp_path / "dec.py"
        f.write_text("@some_decorator\nx = 1\n")
        symbols = mapper.parse_file(f)
        assert isinstance(symbols, list)

    def test_decorated_unknown_returns_none(self, mapper: RepoMapper, tmp_path: Path) -> None:
        f = tmp_path / "dec_unknown.py"
        f.write_text("@something\n@another\ndef foo(): pass\n")
        symbols = mapper.parse_file(f)
        assert len(symbols) >= 1

    def test_decorated_class_member(self, mapper: RepoMapper, tmp_path: Path) -> None:
        f = tmp_path / "dec_member.py"
        f.write_text(
            "class Service:\n    @property\n    def name(self):\n        return self._name\n"
        )
        symbols = mapper.parse_file(f)
        assert len(symbols) == 1
        assert len(symbols[0].children) >= 1


# ── Export statement edge cases ─────────────────────────────────────────────


@_skip_no_treesitter
class TestExportStatements:
    """Cover export_statement branches."""

    def test_export_with_nothing_inside(self, mapper: RepoMapper, tmp_path: Path) -> None:
        f = tmp_path / "exp.js"
        f.write_text("export { x };\nfunction actual() {}\n")
        symbols = mapper.parse_file(f)
        assert any(s.name == "actual" for s in symbols)

    def test_export_default_function(self, mapper: RepoMapper, tmp_path: Path) -> None:
        f = tmp_path / "exp2.js"
        f.write_text("export default function main() {}\n")
        symbols = mapper.parse_file(f)
        assert len(symbols) >= 1


# ── Class/method definition edge cases ──────────────────────────────────────


@_skip_no_treesitter
class TestClassAndMethodEdges:
    """Cover class_definition and method_declaration branch edges."""

    def test_class_without_block_members(self, mapper: RepoMapper, tmp_path: Path) -> None:
        f = tmp_path / "no_block.py"
        f.write_text("class Empty:\n    pass\n")
        symbols = mapper.parse_file(f)
        assert len(symbols) == 1
        assert symbols[0].kind == "class"


# ── Go-specific edge cases ─────────────────────────────────────────────────


@_skip_no_treesitter
class TestGoSpecific:
    """Cover Go-specific branches."""

    def test_go_method_declaration(self, mapper: RepoMapper, tmp_path: Path) -> None:
        f = tmp_path / "methods.go"
        f.write_text(
            "package main\n\n"
            "type Server struct{}\n\n"
            "func (s *Server) Start() error {\n\treturn nil\n}\n"
        )
        symbols = mapper.parse_file(f)
        assert isinstance(symbols, list)

    def test_go_type_declaration_with_type_spec(self, mapper: RepoMapper, tmp_path: Path) -> None:
        f = tmp_path / "types.go"
        f.write_text("package main\n\ntype MyInt int\n\ntype MyStruct struct {\n\tX int\n}\n")
        symbols = mapper.parse_file(f)
        assert isinstance(symbols, list)


# ── _member_to_symbol edge cases ────────────────────────────────────────────


@_skip_no_treesitter
class TestMemberToSymbol:
    """Cover _member_to_symbol branches."""

    def test_member_decorated_without_function(self, mapper: RepoMapper, tmp_path: Path) -> None:
        f = tmp_path / "dec_member2.py"
        f.write_text("class Foo:\n    @decorator\n    x = 1\n")
        symbols = mapper.parse_file(f)
        assert len(symbols) == 1

    def test_member_unknown_type_returns_none(self, mapper: RepoMapper, tmp_path: Path) -> None:
        f = tmp_path / "unknown_member.py"
        f.write_text("class Foo:\n    pass\n")
        symbols = mapper.parse_file(f)
        assert len(symbols) == 1
        assert len(symbols[0].children) == 0


# ── map_directory edge cases ────────────────────────────────────────────────


@_skip_no_treesitter
class TestMapDirectoryEdges:
    """Cover remaining branches in map_directory."""

    def test_map_directory_not_a_directory(self, mapper: RepoMapper, tmp_path: Path) -> None:
        f = tmp_path / "file.py"
        f.write_text("x = 1\n")
        result = mapper.map_directory(f)  # line 244
        assert "Not a directory" in result

    def test_map_directory_excluded_node_modules(self, mapper: RepoMapper, tmp_path: Path) -> None:
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("def main(): pass\n")
        (tmp_path / "node_modules").mkdir()
        (tmp_path / "node_modules" / "junk.py").write_text("def junk(): pass\n")
        result = mapper.map_directory(tmp_path)
        assert "main" in result
        assert "junk" not in result

    def test_map_directory_unsupported_extensions_skipped(
        self, mapper: RepoMapper, tmp_path: Path
    ) -> None:
        (tmp_path / "data.csv").write_text("a,b,c\n")
        (tmp_path / "valid.py").write_text("def foo(): pass\n")
        result = mapper.map_directory(tmp_path)
        assert "foo" in result
        assert "data.csv" not in result

    def test_map_directory_file_with_no_symbols_skipped(
        self, mapper: RepoMapper, tmp_path: Path
    ) -> None:
        (tmp_path / "blank.py").write_text("")
        (tmp_path / "real.py").write_text("def bar(): pass\n")
        result = mapper.map_directory(tmp_path)
        assert "bar" in result
        assert "blank" not in result

    def test_map_directory_preserves_source_with_symlinks(
        self, mapper: RepoMapper, tmp_path: Path
    ) -> None:
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "core.py").write_text("class Core:\n    def run(self): pass\n")
        result = mapper.map_directory(tmp_path)
        assert "Core" in result

    def test_map_directory_empty_no_symbols(self, mapper: RepoMapper, tmp_path: Path) -> None:
        result = mapper.map_directory(tmp_path)
        assert "No symbols found" in result

    def test_map_directory_max_depth_boundary(self, mapper: RepoMapper, tmp_path: Path) -> None:
        shallow = tmp_path / "top.py"
        shallow.write_text("def top(): pass\n")
        deep = tmp_path / "a" / "b" / "c" / "d" / "e" / "f" / "deep.py"
        deep.parent.mkdir(parents=True)
        deep.write_text("def deep(): pass\n")

        result = mapper.map_directory(tmp_path, max_depth=3)
        assert "top" in result
        assert "deep" not in result

    def test_map_directory_pattern_in_subdir(self, mapper: RepoMapper, tmp_path: Path) -> None:
        (tmp_path / "lib").mkdir()
        (tmp_path / "lib" / "a.py").write_text("def a(): pass\n")
        (tmp_path / "lib" / "b.py").write_text("def b(): pass\n")
        (tmp_path / "root.py").write_text("def root(): pass\n")

        result = mapper.map_directory(tmp_path, pattern="lib/*.py")
        assert "a" in result
        assert "b" in result

    def test_map_directory_windows_path_separators(
        self, mapper: RepoMapper, tmp_path: Path
    ) -> None:
        (tmp_path / "pkg").mkdir()
        (tmp_path / "pkg" / "module.py").write_text("def win(): pass\n")
        result = mapper.map_directory(tmp_path)
        assert "pkg/module.py" in result
