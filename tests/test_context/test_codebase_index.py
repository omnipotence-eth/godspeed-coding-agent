"""Tests for codebase indexer and chunker."""

from __future__ import annotations

from pathlib import Path

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
        # Falls back to sliding window
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
