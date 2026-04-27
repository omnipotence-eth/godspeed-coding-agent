"""Tests for UserMemory — SQLite-backed preferences and corrections."""

from __future__ import annotations

from pathlib import Path

import pytest

from godspeed.memory.user_memory import UserMemory


@pytest.fixture
def memory(tmp_path: Path) -> UserMemory:
    """Create a UserMemory with a temp database."""
    mem = UserMemory(db_path=tmp_path / "test_memory.db")
    yield mem
    mem.close()


class TestPreferences:
    """Test preference CRUD operations."""

    def test_set_and_get(self, memory: UserMemory) -> None:
        memory.set("theme", "dark")
        assert memory.get("theme") == "dark"

    def test_get_missing_returns_default(self, memory: UserMemory) -> None:
        assert memory.get("missing") is None
        assert memory.get("missing", "fallback") == "fallback"

    def test_set_overwrites(self, memory: UserMemory) -> None:
        memory.set("model", "gpt-4")
        memory.set("model", "claude-sonnet")
        assert memory.get("model") == "claude-sonnet"

    def test_delete_existing(self, memory: UserMemory) -> None:
        memory.set("key", "value")
        deleted = memory.delete("key")
        assert deleted is True
        assert memory.get("key") is None

    def test_delete_missing(self, memory: UserMemory) -> None:
        deleted = memory.delete("nonexistent")
        assert deleted is False

    def test_list_preferences(self, memory: UserMemory) -> None:
        memory.set("a", "1")
        memory.set("b", "2")
        memory.set("c", "3")
        prefs = memory.list_preferences()
        assert len(prefs) == 3
        assert prefs[0]["key"] == "a"
        assert prefs[2]["key"] == "c"

    def test_list_preferences_empty(self, memory: UserMemory) -> None:
        prefs = memory.list_preferences()
        assert prefs == []

    def test_preference_has_updated_at(self, memory: UserMemory) -> None:
        memory.set("key", "value")
        prefs = memory.list_preferences()
        assert "updated_at" in prefs[0]
        assert isinstance(prefs[0]["updated_at"], float)


class TestCorrections:
    """Test correction recording and retrieval."""

    def test_record_and_get(self, memory: UserMemory) -> None:
        cid = memory.record_correction("print(x)", "logger.info(x)", context="logging")
        corrections = memory.get_corrections()
        assert len(corrections) == 1
        assert corrections[0]["id"] == cid
        assert corrections[0]["original"] == "print(x)"
        assert corrections[0]["corrected"] == "logger.info(x)"
        assert corrections[0]["context"] == "logging"

    def test_corrections_newest_first(self, memory: UserMemory) -> None:
        memory.record_correction("a", "b")
        memory.record_correction("c", "d")
        memory.record_correction("e", "f")
        corrections = memory.get_corrections()
        assert corrections[0]["original"] == "e"
        assert corrections[2]["original"] == "a"

    def test_corrections_limit(self, memory: UserMemory) -> None:
        for i in range(20):
            memory.record_correction(f"old_{i}", f"new_{i}")
        corrections = memory.get_corrections(limit=5)
        assert len(corrections) == 5

    def test_delete_correction(self, memory: UserMemory) -> None:
        cid = memory.record_correction("x", "y")
        assert memory.delete_correction(cid) is True
        assert memory.get_corrections() == []

    def test_delete_correction_missing(self, memory: UserMemory) -> None:
        assert memory.delete_correction(999) is False

    def test_correction_count(self, memory: UserMemory) -> None:
        assert memory.correction_count() == 0
        memory.record_correction("a", "b")
        memory.record_correction("c", "d")
        assert memory.correction_count() == 2

    def test_correction_default_context(self, memory: UserMemory) -> None:
        memory.record_correction("x", "y")
        corrections = memory.get_corrections()
        assert corrections[0]["context"] == ""


class TestDatabaseLifecycle:
    """Test database creation, WAL mode, and concurrent access."""

    def test_creates_db_on_init(self, tmp_path: Path) -> None:
        db_path = tmp_path / "sub" / "dir" / "memory.db"
        mem = UserMemory(db_path=db_path)
        assert db_path.exists()
        mem.close()

    def test_db_path_property(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        mem = UserMemory(db_path=db_path)
        assert mem.db_path == db_path
        mem.close()

    def test_wal_mode_enabled(self, tmp_path: Path) -> None:
        db_path = tmp_path / "wal_test.db"
        mem = UserMemory(db_path=db_path)
        assert mem._conn is not None
        cursor = mem._conn.execute("PRAGMA journal_mode")
        mode = cursor.fetchone()[0]
        assert mode == "wal"
        mem.close()

    def test_concurrent_writes(self, tmp_path: Path) -> None:
        """Two UserMemory instances can write to the same DB safely."""
        db_path = tmp_path / "concurrent.db"
        mem1 = UserMemory(db_path=db_path)
        mem2 = UserMemory(db_path=db_path)

        mem1.set("from_1", "value_1")
        mem2.set("from_2", "value_2")

        assert mem1.get("from_2") == "value_2"
        assert mem2.get("from_1") == "value_1"

        mem1.close()
        mem2.close()

    def test_close_and_reopen(self, tmp_path: Path) -> None:
        db_path = tmp_path / "reopen.db"
        mem = UserMemory(db_path=db_path)
        mem.set("persist", "yes")
        mem.close()

        mem2 = UserMemory(db_path=db_path)
        assert mem2.get("persist") == "yes"
        mem2.close()


class TestEdgeCases:
    """Test edge cases and error paths."""

    def test_default_db_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        mem = UserMemory()
        assert mem.db_path == tmp_path / ".godspeed" / "memory.db"
        mem.close()

    def test_double_close_is_safe(self, tmp_path: Path) -> None:
        mem = UserMemory(db_path=tmp_path / "double.db")
        mem.close()
        mem.close()  # should not raise

    def test_get_after_close_raises(self, tmp_path: Path) -> None:
        mem = UserMemory(db_path=tmp_path / "closed.db")
        mem.close()
        with pytest.raises(RuntimeError, match="closed"):
            mem.get("key")

    def test_set_after_close_raises(self, tmp_path: Path) -> None:
        mem = UserMemory(db_path=tmp_path / "closed.db")
        mem.close()
        with pytest.raises(RuntimeError, match="closed"):
            mem.set("key", "value")
