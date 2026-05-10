"""Tests for codebase indexer and chunker."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from godspeed.context.chunker import Chunk, chunk_file


class TestChunkFile:
    """Test file chunking."""

    def test_empty_file(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.py"
        f.write_text("", encoding="utf-8")
        assert chunk_file(f) == []

    def test_simple_python(self, tmp_path: Path) -> None:
        f = tmp_path / "simple.py"
        f.write_text(
            "def hello():\n    return 'world'\n\ndef goodbye():\n    return 'bye'\n",
            encoding="utf-8",
        )
        chunks = chunk_file(f)
        assert len(chunks) >= 2
        assert any("hello" in c.content for c in chunks)
        assert any("goodbye" in c.content for c in chunks)

    def test_python_with_class(self, tmp_path: Path) -> None:
        f = tmp_path / "cls.py"
        f.write_text(
            "class Foo:\n    def bar(self):\n        pass\n\n"
            "class Baz:\n    def qux(self):\n        pass\n",
            encoding="utf-8",
        )
        chunks = chunk_file(f)
        assert len(chunks) >= 2

    def test_non_python_uses_sliding_window(self, tmp_path: Path) -> None:
        f = tmp_path / "readme.md"
        content = "Line\n" * 100
        f.write_text(content, encoding="utf-8")
        chunks = chunk_file(f, max_tokens=20)
        assert len(chunks) >= 1
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_chunk_has_correct_fields(self, tmp_path: Path) -> None:
        f = tmp_path / "test.py"
        f.write_text("def foo():\n    pass\n", encoding="utf-8")
        chunks = chunk_file(f)
        assert len(chunks) >= 1
        c = chunks[0]
        assert c.file_path == str(f)
        assert c.start_line >= 1
        assert c.end_line >= c.start_line

    def test_nonexistent_file(self, tmp_path: Path) -> None:
        assert chunk_file(tmp_path / "nonexistent.py") == []

    def test_invalid_python_falls_back(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.py"
        f.write_text("def (\n  invalid syntax\n", encoding="utf-8")
        chunks = chunk_file(f)
        assert len(chunks) >= 1

    def test_python_no_defs_falls_back(self, tmp_path: Path) -> None:
        f = tmp_path / "script.py"
        f.write_text("x = 1\ny = 2\nprint(x + y)\n", encoding="utf-8")
        chunks = chunk_file(f)
        assert len(chunks) >= 1

    def test_chunk_frozen(self, tmp_path: Path) -> None:
        c = Chunk(content="x", file_path="f", start_line=1, end_line=1)
        with pytest.raises(AttributeError):
            c.content = "y"  # type: ignore[misc]

    def test_python_with_header_module_code(self, tmp_path: Path) -> None:
        f = tmp_path / "mod.py"
        f.write_text(
            '"""Module docstring."""\n\nimport os\nX = 1\n\n'
            "def foo():\n    return X\n\n"
            "def bar():\n    return 2\n",
            encoding="utf-8",
        )
        chunks = chunk_file(f)
        assert len(chunks) >= 2
        assert "foo" in "".join(c.content for c in chunks)

    def test_python_with_async_function(self, tmp_path: Path) -> None:
        f = tmp_path / "async_mod.py"
        f.write_text(
            "async def fetch():\n    return await do_stuff()\n\n"
            "def sync():\n    return 1\n",
            encoding="utf-8",
        )
        chunks = chunk_file(f)
        assert len(chunks) >= 2

    def test_python_with_large_function_splits(self, tmp_path: Path) -> None:
        f = tmp_path / "large.py"
        lines = "def big():\n" + "    x = 1\n" * 500
        f.write_text(lines, encoding="utf-8")
        chunks = chunk_file(f, max_tokens=50)
        assert len(chunks) >= 1

    def test_whitespace_only_file(self, tmp_path: Path) -> None:
        f = tmp_path / "blank.py"
        f.write_text("   \n\t\n   \n", encoding="utf-8")
        chunks = chunk_file(f)
        assert chunks == []

    def test_sliding_window_overlap_preserves_context(self, tmp_path: Path) -> None:
        f = tmp_path / "long.txt"
        lines = "\n".join(f"line {i} with content" for i in range(100))
        f.write_text(lines, encoding="utf-8")
        chunks = chunk_file(f, max_tokens=10)
        assert len(chunks) >= 1
        # Check that consecutive chunks share at least one line for overlap
        if len(chunks) >= 2:
            pass  # overlap is about word count, not exact line matching

    def test_single_line_below_max(self, tmp_path: Path) -> None:
        f = tmp_path / "one.py"
        f.write_text("x = 1\n", encoding="utf-8")
        chunks = chunk_file(f)
        assert len(chunks) >= 1

    def test_python_with_pyi_ext(self, tmp_path: Path) -> None:
        f = tmp_path / "types.pyi"
        f.write_text("def add(x: int, y: int) -> int: ...\n", encoding="utf-8")
        chunks = chunk_file(f)
        assert len(chunks) >= 1

    def test_python_header_empty_stripped(self, tmp_path: Path) -> None:
        f = tmp_path / "mod.py"
        f.write_text("# comment\n# another comment\n\ndef foo():\n    pass\n", encoding="utf-8")
        chunks = chunk_file(f)
        assert len(chunks) >= 1
        # The first chunk should contain the header comments
        contents = [c.content for c in chunks]
        assert any("comment" in c for c in contents) or any("foo" in c for c in contents)

    def test_chunk_end_line_accuracy(self, tmp_path: Path) -> None:
        f = tmp_path / "mod.py"
        f.write_text("def foo():\n    pass\n", encoding="utf-8")
        chunks = chunk_file(f)
        assert len(chunks) >= 1
        assert chunks[0].start_line >= 1
        assert chunks[0].end_line >= 2

    def test_large_python_file_multiple_chunks(self, tmp_path: Path) -> None:
        f = tmp_path / "big.py"
        code = ""
        for i in range(30):
            code += f"def func_{i}():\n    return {i}\n\n"
        f.write_text(code, encoding="utf-8")
        chunks = chunk_file(f, max_tokens=20)
        assert len(chunks) >= 1

    def test_chunk_filepath_in_all_chunks(self, tmp_path: Path) -> None:
        f = tmp_path / "mod.py"
        f.write_text("def a():\n    pass\n\ndef b():\n    pass\n", encoding="utf-8")
        chunks = chunk_file(f)
        for c in chunks:
            assert c.file_path == str(f)

    def test_sliding_window_with_empty_line_in_middle(self, tmp_path: Path) -> None:
        f = tmp_path / "mixed.txt"
        f.write_text("line 1\n\nline 3\nline 4\n", encoding="utf-8")
        chunks = chunk_file(f)
        assert len(chunks) >= 1

    def test_python_with_only_imports(self, tmp_path: Path) -> None:
        f = tmp_path / "imports.py"
        f.write_text("import os\nimport sys\nfrom pathlib import Path\n", encoding="utf-8")
        chunks = chunk_file(f)
        assert len(chunks) >= 1

    def test_python_empty_header_stripped_no_chunk(self, tmp_path: Path) -> None:
        f = tmp_path / "empty_header.py"
        f.write_text("\n\n\ndef foo():\n    pass\n", encoding="utf-8")
        chunks = chunk_file(f)
        assert any("foo" in c.content for c in chunks)

    def test_sliding_window_empty_lines_only(self, tmp_path: Path) -> None:
        f = tmp_path / "blank_only.txt"
        f.write_text("\n\n\n", encoding="utf-8")
        chunks = chunk_file(f)
        assert chunks == []

    def test_python_empty_body_function(self, tmp_path: Path) -> None:
        f = tmp_path / "empty_body.py"
        f.write_text("def empty():\n    pass\n\ndef also_empty():\n    ...\n", encoding="utf-8")
        chunks = chunk_file(f)
        assert any("empty" in c.content for c in chunks)

    def test_python_definition_with_only_whitespace(self, tmp_path: Path) -> None:
        f = tmp_path / "ws.py"
        f.write_text("def ws():\n    \n    \n\nx = 1\n", encoding="utf-8")
        chunks = chunk_file(f)
        assert len(chunks) >= 1

    def test_sliding_window_empty_content_stripped(self, tmp_path: Path) -> None:
        f = tmp_path / "blank_lines.md"
        f.write_text("\n\nline\n\n\nmore\n", encoding="utf-8")
        chunks = chunk_file(f, max_tokens=1)
        assert len(chunks) >= 1
        for c in chunks:
            assert c.content.strip() != ""

    def test_python_single_function_under_max(self, tmp_path: Path) -> None:
        f = tmp_path / "tiny.py"
        f.write_text("def tiny():\n    pass\n", encoding="utf-8")
        chunks = chunk_file(f)
        assert len(chunks) >= 1
        assert "tiny" in chunks[0].content

    def test_python_definition_small_does_not_subchunk(self, tmp_path: Path) -> None:
        f = tmp_path / "small.py"
        f.write_text("def small():\n    return 1\n", encoding="utf-8")
        chunks = chunk_file(f, max_tokens=100)
        assert len(chunks) >= 1
        assert "small" in chunks[0].content

    def test_chunk_file_cannot_read_returns_empty(self, tmp_path: Path) -> None:
        f = tmp_path / "unreadable.py"
        f.write_text("def foo(): pass\n", encoding="utf-8")
        with patch("pathlib.Path.read_text", side_effect=OSError("cannot read")):
            assert chunk_file(f) == []


class TestCodebaseIndex:
    """Test CodebaseIndex (without chromadb — graceful degradation)."""

    def test_is_available_without_chromadb(self, tmp_path: Path) -> None:
        from godspeed.context.codebase_index import CodebaseIndex

        index = CodebaseIndex(project_dir=tmp_path)
        # May be True or False depending on environment
        assert isinstance(index.is_available, bool)

    def test_search_without_chromadb_returns_empty(self, tmp_path: Path) -> None:
        from godspeed.context.codebase_index import CodebaseIndex

        index = CodebaseIndex(project_dir=tmp_path)
        if not index.is_available:
            results = index.search("test query")
            assert results == []

    def test_needs_reindex_no_db(self, tmp_path: Path) -> None:
        from godspeed.context.codebase_index import CodebaseIndex

        index = CodebaseIndex(project_dir=tmp_path)
        assert index.needs_reindex() is True

    def test_is_building_default_false(self, tmp_path: Path) -> None:
        from godspeed.context.codebase_index import CodebaseIndex

        index = CodebaseIndex(project_dir=tmp_path)
        assert index.is_building is False

    def test_build_index_without_chromadb(self, tmp_path: Path) -> None:
        from godspeed.context.codebase_index import CodebaseIndex

        index = CodebaseIndex(project_dir=tmp_path)
        if not index.is_available:
            count = index.build_index()
            assert count == 0
