"""Tests for SessionMemory — session event logging."""

from __future__ import annotations

from pathlib import Path

import pytest

from godspeed.memory.session import SessionMemory


@pytest.fixture
def session_mem(tmp_path: Path) -> SessionMemory:
    mem = SessionMemory(db_path=tmp_path / "session_test.db")
    yield mem
    mem.close()


class TestSessionLifecycle:
    """Test session start/end/list."""

    def test_start_session(self, session_mem: SessionMemory) -> None:
        session_mem.start_session("s1", "claude-sonnet", project_dir="/home/user/proj")
        s = session_mem.get_session("s1")
        assert s is not None
        assert s["model"] == "claude-sonnet"
        assert s["project_dir"] == "/home/user/proj"

    def test_end_session(self, session_mem: SessionMemory) -> None:
        session_mem.start_session("s1", "gpt-4")
        session_mem.end_session("s1", summary="Fixed bug in auth")
        s = session_mem.get_session("s1")
        assert s is not None
        assert s["ended_at"] is not None
        assert s["summary"] == "Fixed bug in auth"

    def test_get_missing_session(self, session_mem: SessionMemory) -> None:
        assert session_mem.get_session("nonexistent") is None

    def test_list_sessions(self, session_mem: SessionMemory) -> None:
        session_mem.start_session("s1", "m1")
        session_mem.start_session("s2", "m2")
        sessions = session_mem.list_sessions()
        assert len(sessions) == 2

    def test_list_sessions_limit(self, session_mem: SessionMemory) -> None:
        for i in range(10):
            session_mem.start_session(f"s{i}", "m")
        sessions = session_mem.list_sessions(limit=3)
        assert len(sessions) == 3

    def test_duplicate_start_ignored(self, session_mem: SessionMemory) -> None:
        session_mem.start_session("s1", "m1")
        session_mem.start_session("s1", "m2")  # INSERT OR IGNORE
        s = session_mem.get_session("s1")
        assert s is not None
        assert s["model"] == "m1"  # First wins


class TestSessionEvents:
    """Test event recording and retrieval."""

    def test_record_event(self, session_mem: SessionMemory) -> None:
        session_mem.start_session("s1", "m")
        eid = session_mem.record_event("s1", "tool_call", "shell: ls -la")
        assert eid is not None
        assert eid > 0

    def test_get_events(self, session_mem: SessionMemory) -> None:
        session_mem.start_session("s1", "m")
        session_mem.record_event("s1", "tool_call", "file_read: main.py")
        session_mem.record_event("s1", "tool_error", "permission denied")
        events = session_mem.get_events("s1")
        assert len(events) == 2

    def test_get_events_filtered(self, session_mem: SessionMemory) -> None:
        session_mem.start_session("s1", "m")
        session_mem.record_event("s1", "tool_call", "read")
        session_mem.record_event("s1", "tool_error", "fail")
        session_mem.record_event("s1", "tool_call", "write")
        events = session_mem.get_events("s1", event_type="tool_call")
        assert len(events) == 2
        assert all(e["event_type"] == "tool_call" for e in events)

    def test_get_events_limit(self, session_mem: SessionMemory) -> None:
        session_mem.start_session("s1", "m")
        for i in range(20):
            session_mem.record_event("s1", "tool_call", f"action {i}")
        events = session_mem.get_events("s1", limit=5)
        assert len(events) == 5

    def test_event_count(self, session_mem: SessionMemory) -> None:
        session_mem.start_session("s1", "m")
        assert session_mem.event_count("s1") == 0
        session_mem.record_event("s1", "tool_call", "read")
        session_mem.record_event("s1", "tool_call", "write")
        assert session_mem.event_count("s1") == 2

    def test_events_empty_session(self, session_mem: SessionMemory) -> None:
        events = session_mem.get_events("nonexistent")
        assert events == []


class TestDatabaseLifecycle:
    """Test database management."""

    def test_close_and_reopen(self, tmp_path: Path) -> None:
        db_path = tmp_path / "persist.db"
        mem = SessionMemory(db_path=db_path)
        mem.start_session("s1", "m")
        mem.record_event("s1", "test", "hello")
        mem.close()

        mem2 = SessionMemory(db_path=db_path)
        assert mem2.get_session("s1") is not None
        assert mem2.event_count("s1") == 1
        mem2.close()
