"""Tests for godspeed.context.codebase_index."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from godspeed.context.codebase_index import (
    DEFAULT_EXCLUDES,
    INDEXABLE_EXTENSIONS,
    CodebaseIndex,
    SearchResult,
    _is_chromadb_available,
)


class TestSearchResult:
    def test_fields(self):
        r = SearchResult(file_path="main.py", start_line=1, end_line=10, content="code", score=0.95)
        assert r.file_path == "main.py"
        assert r.start_line == 1
        assert r.end_line == 10
        assert r.content == "code"
        assert r.score == 0.95


class TestDefaults:
    def test_excludes_not_empty(self):
        assert len(DEFAULT_EXCLUDES) > 0
        assert "node_modules" in DEFAULT_EXCLUDES
        assert ".venv" in DEFAULT_EXCLUDES

    def test_indexable_extensions(self):
        assert ".py" in INDEXABLE_EXTENSIONS
        assert ".js" in INDEXABLE_EXTENSIONS
        assert ".go" in INDEXABLE_EXTENSIONS


class TestIsChromadbAvailable:
    def test_cached_true(self):
        with patch("godspeed.context.codebase_index._chromadb_available", True):
            assert _is_chromadb_available() is True

    def test_cached_false(self):
        with patch("godspeed.context.codebase_index._chromadb_available", False):
            assert _is_chromadb_available() is False

    def test_import_success(self):
        with patch("godspeed.context.codebase_index._chromadb_available", None):
            with patch("builtins.__import__", return_value=MagicMock()):
                # Reset the cached value
                import godspeed.context.codebase_index as mod

                mod._chromadb_available = None
                result = _is_chromadb_available()
                assert result is True

    def test_import_failure(self):
        # Reset the cached value and make import fail
        import godspeed.context.codebase_index as mod

        mod._chromadb_available = None
        with patch("builtins.__import__", side_effect=ImportError("no chromadb")):
            result = _is_chromadb_available()
            assert result is False


class TestCodebaseIndexInit:
    def test_init_default_db_path(self):
        idx = CodebaseIndex(project_dir=Path("/tmp/test"))  # noqa: S108
        assert idx._db_path == Path("/tmp/test") / ".godspeed" / "index" / "chroma"  # noqa: S108

    def test_init_custom_db_path(self):
        custom = Path("/custom/db")
        idx = CodebaseIndex(project_dir=Path("/tmp/test"), db_path=custom)  # noqa: S108
        assert idx._db_path == custom

    def test_is_available(self):
        with patch("godspeed.context.codebase_index._is_chromadb_available", return_value=True):
            idx = CodebaseIndex(project_dir=Path("/tmp/test"))  # noqa: S108
            assert idx.is_available is True

    def test_is_not_available(self):
        with patch("godspeed.context.codebase_index._is_chromadb_available", return_value=False):
            idx = CodebaseIndex(project_dir=Path("/tmp/test"))  # noqa: S108
            assert idx.is_available is False

    def test_is_building_default(self):
        idx = CodebaseIndex(project_dir=Path("/tmp/test"))  # noqa: S108
        assert idx.is_building is False

    def test_close(self):
        idx = CodebaseIndex(project_dir=Path("/tmp/test"))  # noqa: S108
        idx._client = MagicMock()
        idx._collection = MagicMock()
        idx.close()
        assert idx._collection is None
        assert idx._client is None


class TestCodebaseIndexBuild:
    def test_build_index_chromadb_unavailable(self):
        idx = CodebaseIndex(project_dir=Path("/tmp/test"))  # noqa: S108
        with patch("godspeed.context.codebase_index._is_chromadb_available", return_value=False):
            count = idx.build_index()
            assert count == 0

    def test_build_index_no_files(self, tmp_path):
        idx = CodebaseIndex(project_dir=tmp_path)
        with patch("godspeed.context.codebase_index._is_chromadb_available", return_value=True):
            mock_collection = MagicMock()
            mock_collection.count.return_value = 0
            with patch.object(idx, "_iter_files", return_value=[]):
                with patch.object(idx, "_ensure_collection", return_value=mock_collection):
                    count = idx.build_index()
                    assert count == 0

    def test_build_index_with_files(self, tmp_path):
        # Create a test file
        test_file = tmp_path / "main.py"
        test_file.write_text("def hello():\n    pass\n")

        from godspeed.context.chunker import Chunk

        idx = CodebaseIndex(project_dir=tmp_path)
        with patch("godspeed.context.codebase_index._is_chromadb_available", return_value=True):
            mock_collection = MagicMock()
            mock_collection.count.return_value = 5
            with patch.object(idx, "_ensure_collection", return_value=mock_collection):
                with patch.object(idx, "_iter_files", return_value=[test_file]):
                    with patch("godspeed.context.codebase_index.chunk_file") as mock_chunk:
                        mock_chunk.return_value = [
                            Chunk(
                                content="chunk1", file_path=str(test_file), start_line=1, end_line=2
                            ),
                            Chunk(
                                content="chunk2", file_path=str(test_file), start_line=3, end_line=4
                            ),
                        ]
                        count = idx.build_index()
                        assert count == 2

    def test_needs_reindex_no_client(self):
        idx = CodebaseIndex(project_dir=Path("/tmp/test"))  # noqa: S108
        # No client and no index_time -> needs reindex
        assert idx.needs_reindex() is True

    def test_needs_reindex_with_client(self):
        idx = CodebaseIndex(project_dir=Path("/tmp/test"))  # noqa: S108
        idx._client = MagicMock()
        idx._index_time = 100.0
        # collection exists but is empty -> needs reindex
        mock_collection = MagicMock()
        mock_collection.count.return_value = 5
        idx._client.get_collection.return_value = mock_collection
        # Mock _ensure_collection to return the mock collection
        with patch.object(idx, "_ensure_collection", return_value=mock_collection):
            assert idx.needs_reindex() is False  # has data, no newer files


class TestCodebaseIndexSearch:
    def test_search_not_available(self):
        idx = CodebaseIndex(project_dir=Path("/tmp/test"))  # noqa: S108
        with patch("godspeed.context.codebase_index._is_chromadb_available", return_value=False):
            results = idx.search("query")
            assert results == []

    def test_search_no_collection(self):
        idx = CodebaseIndex(project_dir=Path("/tmp/test"))  # noqa: S108
        idx._client = MagicMock()
        idx._collection = None
        results = idx.search("query")
        assert results == []

    def test_search_with_results(self):
        idx = CodebaseIndex(project_dir=Path("/tmp/test"))  # noqa: S108
        idx._collection = MagicMock()
        idx._collection.count.return_value = 5
        idx._collection.query.return_value = {
            "ids": [["1", "2"]],
            "documents": [["code1", "code2"]],
            "metadatas": [
                [
                    {"file_path": "a.py", "start_line": 1, "end_line": 5},
                    {"file_path": "b.py", "start_line": 10, "end_line": 20},
                ]
            ],
            "distances": [[0.1, 0.3]],
        }
        with patch("godspeed.context.codebase_index._is_chromadb_available", return_value=True):
            results = idx.search("query", top_k=2)
            assert len(results) == 2
            assert results[0].file_path == "a.py"
            assert results[0].score == 0.9  # 1.0 - 0.1
            assert results[1].file_path == "b.py"

    def test_search_empty_results(self):
        idx = CodebaseIndex(project_dir=Path("/tmp/test"))  # noqa: S108
        idx._collection = MagicMock()
        idx._collection.query.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }
        results = idx.search("query")
        assert results == []


class TestIterFiles:
    def test_iter_files_excludes(self, tmp_path):
        idx = CodebaseIndex(project_dir=tmp_path)
        # Create excluded dir
        (tmp_path / "node_modules").mkdir()
        (tmp_path / "node_modules" / "x.js").write_text("x")
        # Create included file
        (tmp_path / "main.py").write_text("code")

        files = idx._iter_files(excludes={"node_modules"})
        paths = [str(f) for f in files]
        assert all("node_modules" not in p for p in paths)
        assert any("main.py" in p for p in paths)

    def test_iter_files_extensions(self, tmp_path):
        idx = CodebaseIndex(project_dir=tmp_path)
        (tmp_path / "main.py").write_text("code")
        (tmp_path / "readme.md").write_text("# readme")

        files = list(idx._iter_files(excludes=set()))
        paths = [str(f) for f in files]
        assert any("main.py" in p for p in paths)
        # .md is in INDEXABLE_EXTENSIONS, so readme.md should be included
        assert any("readme.md" in p for p in paths)

    def test_iter_files_nonexistent_dir(self, tmp_path):
        idx = CodebaseIndex(project_dir=tmp_path)
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("code")
        # Remove src to simulate missing
        import shutil

        shutil.rmtree(tmp_path / "src")
        files = list(idx._iter_files(excludes=set()))
        assert files == []  # Should not crash


class TestBuildIndexAsync:
    @pytest.mark.asyncio
    async def test_build_index_async(self):
        idx = CodebaseIndex(project_dir=Path("/tmp/test"))  # noqa: S108
        with patch.object(idx, "build_index", return_value=5) as mock_build:
            result = await idx.build_index_async()
            assert result == 5
            mock_build.assert_called_once()
