"""Coverage gap tests for codebase_index — all missed branches."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from godspeed.context.chunker import Chunk
from godspeed.context.codebase_index import (
    CodebaseIndex,
    _is_chromadb_available,
)


# ── _is_chromadb_available — actual import path ─────────────────────────────


class TestIsChromadbAvailableCoverage:
    """Cover the actual _is_chromadb_available import success/failure paths."""

    def test_import_failure_path(self):
        import sys

        mod = sys.modules["godspeed.context.codebase_index"]
        mod._chromadb_available = None
        result = _is_chromadb_available()
        assert isinstance(result, bool)


# ── _load_mtimes / _save_mtimes — error path ────────────────────────────────


class TestMtimesHandling:
    """Cover _load_mtimes error path and _save_mtimes."""

    def test_load_mtimes_corrupt_json_returns_empty(self, tmp_path):
        idx = CodebaseIndex(project_dir=tmp_path)
        mtimes_path = idx._mtimes_path()
        mtimes_path.parent.mkdir(parents=True, exist_ok=True)
        mtimes_path.write_text("not valid json {{{")

        result = idx._load_mtimes()
        assert result == {}

    def test_load_mtimes_no_file_returns_empty(self, tmp_path):
        idx = CodebaseIndex(project_dir=tmp_path)
        result = idx._load_mtimes()
        assert result == {}

    def test_save_and_load_mtimes(self, tmp_path):
        idx = CodebaseIndex(project_dir=tmp_path)
        data = {"/test/file.py": 100.0, "/test/other.py": 200.0}
        idx._save_mtimes(data)
        loaded = idx._load_mtimes()
        assert loaded == data


# ── _add_batch in build_index ──────────────────────────────────────────────


class TestBuildIndexBatch:
    """Cover _add_batch calls within build_index loop."""

    def test_build_index_adds_batches(self, tmp_path):
        test_file = tmp_path / "main.py"
        test_file.write_text("def foo():\n    pass\n\ndef bar():\n    pass\n")

        idx = CodebaseIndex(project_dir=tmp_path)
        mock_collection = MagicMock()
        mock_collection.count.return_value = 5  # existing data

        with patch("godspeed.context.codebase_index._is_chromadb_available", return_value=True):
            with patch.object(idx, "_ensure_collection", return_value=mock_collection):
                with patch.object(idx, "_iter_files", return_value=[test_file]):
                    with patch("godspeed.context.codebase_index.chunk_file") as mock_chunk:
                        chunks = [
                            Chunk(
                                content=f"chunk_{i}",
                                file_path=str(test_file),
                                start_line=i * 2 + 1,
                                end_line=i * 2 + 2,
                            )
                            for i in range(250)
                        ]
                        mock_chunk.return_value = chunks
                        count = idx.build_index()
                        # 250 chunks with batch_size=100 → 3 batches
                        assert count == 250
                        assert mock_collection.upsert.call_count == 3
                        mock_collection.delete.assert_called_once_with(where={"indexed": True})

    def test_build_index_single_batch_no_delete(self, tmp_path):
        test_file = tmp_path / "main.py"
        test_file.write_text("x = 1\n")

        idx = CodebaseIndex(project_dir=tmp_path)
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0  # no existing data

        with patch("godspeed.context.codebase_index._is_chromadb_available", return_value=True):
            with patch.object(idx, "_ensure_collection", return_value=mock_collection):
                with patch.object(idx, "_iter_files", return_value=[test_file]):
                    with patch("godspeed.context.codebase_index.chunk_file") as mock_chunk:
                        mock_chunk.return_value = [
                            Chunk(content="c1", file_path=str(test_file), start_line=1, end_line=2)
                        ]
                        count = idx.build_index()
                        assert count == 1
                        mock_collection.delete.assert_not_called()

    def test_build_index_clears_building_on_error(self, tmp_path):
        test_file = tmp_path / "main.py"
        test_file.write_text("def foo(): pass\n")

        idx = CodebaseIndex(project_dir=tmp_path)
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0

        with patch("godspeed.context.codebase_index._is_chromadb_available", return_value=True):
            with patch.object(idx, "_ensure_collection", return_value=mock_collection):
                with patch.object(idx, "_iter_files", return_value=[test_file]):
                    with patch(
                        "godspeed.context.codebase_index.chunk_file",
                        side_effect=RuntimeError("chunk fail"),
                    ):
                        try:
                            idx.build_index()
                        except RuntimeError:
                            pass
                        assert idx.is_building is False

    def test_build_index_building_flag(self, tmp_path):
        idx = CodebaseIndex(project_dir=tmp_path)
        assert idx.is_building is False

        test_file = tmp_path / "main.py"
        test_file.write_text("x = 1\n")

        mock_collection = MagicMock()
        mock_collection.count.return_value = 0

        with patch("godspeed.context.codebase_index._is_chromadb_available", return_value=True):
            with patch.object(idx, "_ensure_collection", return_value=mock_collection):
                with patch.object(idx, "_iter_files", return_value=[test_file]):
                    with patch("godspeed.context.codebase_index.chunk_file") as mock_chunk:
                        mock_chunk.return_value = [
                            Chunk(content="c", file_path=str(test_file), start_line=1, end_line=2)
                        ]
                        idx.build_index()
                    # building flag should be set to False via finally
                    assert idx.is_building is False


# ── update_index — incremental update ───────────────────────────────────────


class TestUpdateIndex:
    """Cover update_index and update_index_async."""

    def test_update_index_not_available(self, tmp_path):
        idx = CodebaseIndex(project_dir=tmp_path)
        with patch("godspeed.context.codebase_index._is_chromadb_available", return_value=False):
            result = idx.update_index()
            assert result == 0

    def test_update_index_new_file(self, tmp_path):
        test_file = tmp_path / "new_file.py"
        test_file.write_text("def foo(): pass\n")

        idx = CodebaseIndex(project_dir=tmp_path)
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0

        with patch("godspeed.context.codebase_index._is_chromadb_available", return_value=True):
            with patch.object(idx, "_ensure_collection", return_value=mock_collection):
                with patch.object(idx, "_load_mtimes", return_value={}):
                    with patch.object(idx, "_iter_files", return_value=[test_file]):
                        with patch("godspeed.context.codebase_index.chunk_file") as mock_chunk:
                            mock_chunk.return_value = [
                                Chunk(
                                    content="c1", file_path=str(test_file), start_line=1, end_line=2
                                ),
                                Chunk(
                                    content="c2", file_path=str(test_file), start_line=3, end_line=4
                                ),
                            ]
                            result = idx.update_index()
                            assert result == 2

    def test_update_index_modified_file(self, tmp_path):
        test_file = tmp_path / "mod.py"
        test_file.write_text("def foo(): pass\n")

        idx = CodebaseIndex(project_dir=tmp_path)
        mock_collection = MagicMock()

        old_mtime = test_file.stat().st_mtime - 10  # older

        with patch("godspeed.context.codebase_index._is_chromadb_available", return_value=True):
            with patch.object(idx, "_ensure_collection", return_value=mock_collection):
                with patch.object(idx, "_load_mtimes", return_value={str(test_file): old_mtime}):
                    with patch.object(idx, "_iter_files", return_value=[test_file]):
                        with patch("godspeed.context.codebase_index.chunk_file") as mock_chunk:
                            mock_chunk.return_value = [
                                Chunk(
                                    content="c", file_path=str(test_file), start_line=1, end_line=2
                                )
                            ]
                            result = idx.update_index()
                            assert result > 0

    def test_update_index_deleted_file(self, tmp_path):
        idx = CodebaseIndex(project_dir=tmp_path)
        mock_collection = MagicMock()

        with patch("godspeed.context.codebase_index._is_chromadb_available", return_value=True):
            with patch.object(idx, "_ensure_collection", return_value=mock_collection):
                with patch.object(idx, "_load_mtimes", return_value={"/deleted.py": 100.0}):
                    with patch.object(idx, "_iter_files", return_value=[]):
                        result = idx.update_index()
                        # deleted file removed, no new chunks
                        assert result == 0
                        mock_collection.delete.assert_called_once()

    def test_update_index_up_to_date(self, tmp_path):
        test_file = tmp_path / "fresh.py"
        test_file.write_text("def foo(): pass\n")

        idx = CodebaseIndex(project_dir=tmp_path)
        mock_collection = MagicMock()

        current_mtime = test_file.stat().st_mtime

        with patch("godspeed.context.codebase_index._is_chromadb_available", return_value=True):
            with patch.object(idx, "_ensure_collection", return_value=mock_collection):
                with patch.object(
                    idx, "_load_mtimes", return_value={str(test_file): current_mtime}
                ):
                    with patch.object(idx, "_iter_files", return_value=[test_file]):
                        result = idx.update_index()
                        assert result == 0  # up to date

    def test_update_index_clears_building(self, tmp_path):
        idx = CodebaseIndex(project_dir=tmp_path)
        with patch("godspeed.context.codebase_index._is_chromadb_available", return_value=True):
            with patch.object(idx, "_ensure_collection", side_effect=RuntimeError("fail")):
                try:
                    idx.update_index()
                except RuntimeError:
                    pass
                assert idx.is_building is False

    @pytest.mark.asyncio
    async def test_update_index_async(self):
        idx = CodebaseIndex(project_dir=Path("/tmp/test"))
        with patch.object(idx, "update_index", return_value=10) as mock_update:
            result = await idx.update_index_async()
            assert result == 10
            mock_update.assert_called_once()


# ── Search edge cases ───────────────────────────────────────────────────────


class TestSearchEdgeCases:
    """Cover search edge branches."""

    def test_search_while_building_returns_empty(self):
        idx = CodebaseIndex(project_dir=Path("/tmp/test"))
        idx._building = True
        with patch("godspeed.context.codebase_index._is_chromadb_available", return_value=True):
            results = idx.search("query")
            assert results == []

    def test_search_empty_collection_returns_empty(self):
        idx = CodebaseIndex(project_dir=Path("/tmp/test"))
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        idx._collection = mock_collection

        with patch("godspeed.context.codebase_index._is_chromadb_available", return_value=True):
            results = idx.search("query")
            assert results == []

    def test_search_with_empty_query_returns_empty(self):
        idx = CodebaseIndex(project_dir=Path("/tmp/test"))
        with patch("godspeed.context.codebase_index._is_chromadb_available", return_value=True):
            mock_collection = MagicMock()
            mock_collection.count.return_value = 3
            with patch.object(idx, "_ensure_collection", return_value=mock_collection):
                results = idx.search("")
                assert results == []

    def test_search_exception_returns_empty(self):
        idx = CodebaseIndex(project_dir=Path("/tmp/test"))
        mock_collection = MagicMock()
        mock_collection.query.side_effect = RuntimeError("query crash")

        with patch("godspeed.context.codebase_index._is_chromadb_available", return_value=True):
            with patch.object(idx, "_ensure_collection", return_value=mock_collection):
                mock_collection.count.return_value = 3
                results = idx.search("query")
                assert results == []

    def test_search_negative_distance_clamped(self):
        idx = CodebaseIndex(project_dir=Path("/tmp/test"))
        mock_collection = MagicMock()
        mock_collection.count.return_value = 3
        mock_collection.query.return_value = {
            "ids": [["1"]],
            "documents": [["code1"]],
            "metadatas": [[{"file_path": "a.py", "start_line": 1, "end_line": 5}]],
            "distances": [[2.5]],  # > 1.0 → score = 0.0
        }

        with patch("godspeed.context.codebase_index._is_chromadb_available", return_value=True):
            with patch.object(idx, "_ensure_collection", return_value=mock_collection):
                results = idx.search("query")
                assert len(results) == 1
                assert results[0].score == 0.0


# ── needs_reindex — all branches ────────────────────────────────────────────


class TestNeedsReindex:
    """Cover all branches in needs_reindex."""

    def test_no_index_time_no_db_path(self, tmp_path):
        idx = CodebaseIndex(project_dir=tmp_path, db_path=tmp_path / "nonexistent" / "chroma")
        assert idx.needs_reindex() is True

    def test_no_index_time_with_db_collection_empty(self, tmp_path):
        idx = CodebaseIndex(project_dir=tmp_path)
        idx._index_time = None
        db_path = tmp_path / ".godspeed" / "index" / "chroma"
        db_path.mkdir(parents=True)

        mock_collection = MagicMock()
        mock_collection.count.return_value = 0

        with patch("godspeed.context.codebase_index._is_chromadb_available", return_value=True):
            with patch.object(idx, "_ensure_collection", return_value=mock_collection):
                assert idx.needs_reindex() is True

    def test_no_index_time_collection_not_empty(self, tmp_path):
        idx = CodebaseIndex(project_dir=tmp_path)
        idx._index_time = None
        db_path = tmp_path / ".godspeed" / "index" / "chroma"
        db_path.mkdir(parents=True)

        mock_collection = MagicMock()
        mock_collection.count.return_value = 10

        with patch("godspeed.context.codebase_index._is_chromadb_available", return_value=True):
            with patch.object(idx, "_ensure_collection", return_value=mock_collection):
                assert idx.needs_reindex() is False  # has data

    def test_no_index_time_ensure_collection_fails(self, tmp_path):
        idx = CodebaseIndex(project_dir=tmp_path)
        idx._index_time = None

        with patch("godspeed.context.codebase_index._is_chromadb_available", return_value=True):
            with patch.object(idx, "_ensure_collection", side_effect=Exception("db fail")):
                assert idx.needs_reindex() is True

    def test_no_index_time_not_available(self, tmp_path):
        idx = CodebaseIndex(project_dir=tmp_path)
        idx._index_time = None

        with patch("godspeed.context.codebase_index._is_chromadb_available", return_value=False):
            assert idx.needs_reindex() is True

    def test_empty_mtimes_fallback_scan(self, tmp_path):
        test_file = tmp_path / "new.py"
        test_file.write_text("x = 1\n")
        idx = CodebaseIndex(project_dir=tmp_path)
        idx._index_time = 0.0  # very old

        with patch.object(idx, "_load_mtimes", return_value={}):
            assert idx.needs_reindex() is True

    def test_empty_mtimes_no_new_files(self, tmp_path):
        idx = CodebaseIndex(project_dir=tmp_path)
        idx._index_time = time.time() + 1000  # future

        with patch.object(idx, "_load_mtimes", return_value={}):
            with patch.object(idx, "_iter_files", return_value=[]):
                assert idx.needs_reindex() is False

    def test_empty_mtimes_oserror_fallback(self, tmp_path):
        idx = CodebaseIndex(project_dir=tmp_path)
        idx._index_time = 0.0

        fake_file = MagicMock()
        fake_file.stat.side_effect = OSError("cannot stat")

        with patch.object(idx, "_load_mtimes", return_value={}):
            with patch.object(idx, "_iter_files", return_value=[fake_file]):
                assert idx.needs_reindex() is False

    def test_mtime_mismatch_detects_new_file(self, tmp_path):
        test_file = tmp_path / "new.py"
        test_file.write_text("x = 1\n")
        idx = CodebaseIndex(project_dir=tmp_path)
        idx._index_time = 0.0  # very old — before file creation
        db_path = tmp_path / ".godspeed" / "index" / "chroma"
        db_path.mkdir(parents=True)

        with patch.object(idx, "_load_mtimes", return_value={}):
            with patch.object(idx, "_iter_files", return_value=[test_file]):
                assert idx.needs_reindex() is True  # file not in old mtimes


# ── get_stats ───────────────────────────────────────────────────────────────


class TestGetStats:
    """Cover get_stats all branches."""

    def test_get_stats_available(self):
        idx = CodebaseIndex(project_dir=Path("/tmp/test"))
        mock_collection = MagicMock()
        mock_collection.count.return_value = 42

        with patch("godspeed.context.codebase_index._is_chromadb_available", return_value=True):
            idx._collection = mock_collection
            stats = idx.get_stats()
            assert stats == {"available": True, "count": 42}

    def test_get_stats_exception(self):
        idx = CodebaseIndex(project_dir=Path("/tmp/test"))
        with patch("godspeed.context.codebase_index._is_chromadb_available", return_value=True):
            with patch.object(idx, "_ensure_collection", side_effect=Exception("fail")):
                stats = idx.get_stats()
                assert stats == {"available": False, "count": 0}


# ── clear ───────────────────────────────────────────────────────────────────


class TestClear:
    """Cover clear all branches."""

    def test_clear_success(self):
        idx = CodebaseIndex(project_dir=Path("/tmp/test"))
        mock_collection = MagicMock()
        mock_collection.count.return_value = 10

        with patch("godspeed.context.codebase_index._is_chromadb_available", return_value=True):
            idx._collection = mock_collection
            result = idx.clear()
            assert result is True
            mock_collection.delete.assert_called_once_with(where={"indexed": True})

    def test_clear_empty_collection(self):
        idx = CodebaseIndex(project_dir=Path("/tmp/test"))
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0

        with patch("godspeed.context.codebase_index._is_chromadb_available", return_value=True):
            idx._collection = mock_collection
            result = idx.clear()
            assert result is True
            mock_collection.delete.assert_not_called()

    def test_clear_exception(self):
        idx = CodebaseIndex(project_dir=Path("/tmp/test"))
        mock_collection = MagicMock()
        mock_collection.count.side_effect = RuntimeError("count fail")

        with patch("godspeed.context.codebase_index._is_chromadb_available", return_value=True):
            idx._collection = mock_collection
            result = idx.clear()
            assert result is False


# ── build (async wrapper) ──────────────────────────────────────────────────


class TestBuildAsync:
    """Cover build async wrapper."""

    @pytest.mark.asyncio
    async def test_build_available_returns_true(self):
        idx = CodebaseIndex(project_dir=Path("/tmp/test"))
        with patch("godspeed.context.codebase_index._is_chromadb_available", return_value=True):
            with patch.object(idx, "build_index_async", return_value=5):
                result = await idx.build()
                assert result is True

    @pytest.mark.asyncio
    async def test_build_available_returns_false_when_zero(self):
        idx = CodebaseIndex(project_dir=Path("/tmp/test"))
        with patch("godspeed.context.codebase_index._is_chromadb_available", return_value=True):
            with patch.object(idx, "build_index_async", return_value=0):
                result = await idx.build()
                assert result is False


# ── add_file / remove_file / reindex_file with chromadb ────────────────────


class TestFileOperations:
    """Cover add_file, remove_file, reindex_file with available chromadb."""

    def test_add_file_success(self, tmp_path):
        test_file = tmp_path / "add.py"
        test_file.write_text("def foo(): pass\n")

        idx = CodebaseIndex(project_dir=tmp_path)
        mock_collection = MagicMock()

        with patch("godspeed.context.codebase_index._is_chromadb_available", return_value=True):
            with patch.object(idx, "_ensure_collection", return_value=mock_collection):
                with patch("godspeed.context.codebase_index.chunk_file") as mock_chunk:
                    mock_chunk.return_value = [
                        Chunk(content="c1", file_path=str(test_file), start_line=1, end_line=2)
                    ]
                    result = idx.add_file(test_file)
                    assert result is True
                    mock_collection.upsert.assert_called_once()

    def test_add_file_empty_chunks(self, tmp_path):
        test_file = tmp_path / "empty.py"
        test_file.write_text("")

        idx = CodebaseIndex(project_dir=tmp_path)
        mock_collection = MagicMock()

        with patch("godspeed.context.codebase_index._is_chromadb_available", return_value=True):
            with patch.object(idx, "_ensure_collection", return_value=mock_collection):
                with patch("godspeed.context.codebase_index.chunk_file", return_value=[]):
                    result = idx.add_file(test_file)
                    assert result is True  # no chunks, still success

    def test_add_file_exception(self, tmp_path):
        test_file = tmp_path / "fail.py"
        test_file.write_text("x = 1\n")

        idx = CodebaseIndex(project_dir=tmp_path)

        with patch("godspeed.context.codebase_index._is_chromadb_available", return_value=True):
            with patch.object(idx, "_ensure_collection", side_effect=RuntimeError("fail")):
                result = idx.add_file(test_file)
                assert result is False

    def test_remove_file_success(self, tmp_path):
        idx = CodebaseIndex(project_dir=tmp_path)
        mock_collection = MagicMock()

        with patch("godspeed.context.codebase_index._is_chromadb_available", return_value=True):
            with patch.object(idx, "_ensure_collection", return_value=mock_collection):
                result = idx.remove_file(Path("/tmp/test/file.py"))
                assert result is True
                mock_collection.delete.assert_called_once_with(
                    where={"file_path": str(Path("/tmp/test/file.py"))}
                )

    def test_remove_file_exception(self):
        idx = CodebaseIndex(project_dir=Path("/tmp/test"))

        with patch("godspeed.context.codebase_index._is_chromadb_available", return_value=True):
            with patch.object(idx, "_ensure_collection", side_effect=Exception("fail")):
                result = idx.remove_file(Path("/tmp/test/file.py"))
                assert result is False

    def test_reindex_file_success(self, tmp_path):
        test_file = tmp_path / "reidx.py"
        test_file.write_text("def foo(): pass\n")

        idx = CodebaseIndex(project_dir=tmp_path)
        mock_collection = MagicMock()

        with patch("godspeed.context.codebase_index._is_chromadb_available", return_value=True):
            with patch.object(idx, "_ensure_collection", return_value=mock_collection):
                with patch("godspeed.context.codebase_index.chunk_file") as mock_chunk:
                    mock_chunk.return_value = [
                        Chunk(content="c", file_path=str(test_file), start_line=1, end_line=2)
                    ]
                    result = idx.reindex_file(test_file)
                    assert result is True

    def test_reindex_file_not_available(self, tmp_path):
        idx = CodebaseIndex(project_dir=tmp_path)
        with patch("godspeed.context.codebase_index._is_chromadb_available", return_value=False):
            result = idx.reindex_file(tmp_path / "test.py")
            assert result is False


# ── _iter_files — PermissionError / OSError ────────────────────────────────


class TestIterFilesErrors:
    """Cover PermissionError/OSError in _iter_files."""

    def test_permission_error_skips_directory(self, tmp_path):
        (tmp_path / "allowed.py").write_text("x = 1\n")
        idx = CodebaseIndex(project_dir=tmp_path)

        original_iterdir = Path.iterdir
        call_count = [0]

        def mock_iterdir(self_obj):
            call_count[0] += 1
            if call_count[0] == 1:
                real = original_iterdir(self_obj)
                results = []
                for entry in real:
                    if entry.is_dir():
                        raise PermissionError("denied")
                    results.append(entry)
            else:
                results = list(original_iterdir(self_obj))
            return results

        with patch.object(Path, "iterdir", mock_iterdir):
            files = list(idx._iter_files(excludes=set()))
            assert any("allowed.py" in str(f) for f in files)

    def test_oserror_skips_directory(self, tmp_path):
        (tmp_path / "safe.py").write_text("pass\n")
        idx = CodebaseIndex(project_dir=tmp_path)

        original_iterdir = Path.iterdir
        call_count = [0]

        def mock_iterdir_oserror(self_obj):
            call_count[0] += 1
            if call_count[0] == 1:
                real = original_iterdir(self_obj)
                results = []
                for entry in real:
                    if entry.is_dir():
                        raise OSError("cannot access")
                    results.append(entry)
            else:
                results = list(original_iterdir(self_obj))
            return results

        with patch.object(Path, "iterdir", mock_iterdir_oserror):
            files = list(idx._iter_files(excludes=set()))
            assert any("safe.py" in str(f) for f in files)


# ── Ensure collection on first call ─────────────────────────────────────────


class TestEnsureCollection:
    """Cover _ensure_collection creating chromadb client."""

    def test_ensure_collection_creates_client(self, tmp_path):
        idx = CodebaseIndex(project_dir=tmp_path)

        mock_chromadb = MagicMock()
        mock_client = MagicMock()
        mock_collection = MagicMock()

        mock_chromadb.PersistentClient.return_value = mock_client
        mock_client.get_or_create_collection.return_value = mock_collection

        with patch.dict("sys.modules", {"chromadb": mock_chromadb}):
            with patch("godspeed.context.codebase_index._is_chromadb_available", return_value=True):
                result = idx._ensure_collection()
                assert result is mock_collection
                assert idx._client is mock_client
                mock_chromadb.PersistentClient.assert_called_once()

    def test_ensure_collection_cached(self, tmp_path):
        idx = CodebaseIndex(project_dir=tmp_path)
        mock_collection = MagicMock()
        idx._collection = mock_collection

        result = idx._ensure_collection()
        assert result is mock_collection
