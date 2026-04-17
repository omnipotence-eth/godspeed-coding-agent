"""Tests for the auto-index helper (v2.9.0)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from godspeed.context.auto_index import _run_auto_index, maybe_start_auto_index


def test_disabled_returns_none(tmp_path: Path) -> None:
    """auto_index_enabled=False short-circuits."""
    assert maybe_start_auto_index(tmp_path, auto_index_enabled=False) is None


def test_skips_when_chromadb_unavailable(tmp_path: Path) -> None:
    """When chromadb isn't installed, is_available=False → skip."""
    fake_index = MagicMock(is_available=False)
    with patch("godspeed.context.codebase_index.CodebaseIndex", return_value=fake_index):
        result = maybe_start_auto_index(tmp_path, auto_index_enabled=True)
    assert result is None


def test_skips_when_index_fresh(tmp_path: Path) -> None:
    """Fresh index → no rebuild scheduled."""
    fake_index = MagicMock(is_available=True)
    fake_index.needs_reindex.return_value = False
    with patch("godspeed.context.codebase_index.CodebaseIndex", return_value=fake_index):
        result = maybe_start_auto_index(tmp_path, auto_index_enabled=True)
    assert result is None


@pytest.mark.asyncio
async def test_schedules_task_when_stale(tmp_path: Path) -> None:
    """Stale index → a Task is returned + build_index_async is scheduled."""
    fake_index = MagicMock(is_available=True)
    fake_index.needs_reindex.return_value = True
    fake_index.build_index_async = AsyncMock(return_value=42)
    with patch("godspeed.context.codebase_index.CodebaseIndex", return_value=fake_index):
        task = maybe_start_auto_index(tmp_path, auto_index_enabled=True)
    assert isinstance(task, asyncio.Task)
    result = await task
    assert result == 42
    fake_index.build_index_async.assert_awaited_once()


@pytest.mark.asyncio
async def test_exception_during_build_is_swallowed(tmp_path: Path) -> None:
    """A failing build_index_async must not crash the session."""
    fake_index = MagicMock()
    fake_index.build_index_async = AsyncMock(side_effect=RuntimeError("disk full"))
    result = await _run_auto_index(fake_index)
    assert result == 0


def test_codebase_index_constructor_failure_returns_none(tmp_path: Path) -> None:
    """If instantiating CodebaseIndex raises, we swallow + return None."""
    with patch(
        "godspeed.context.codebase_index.CodebaseIndex",
        side_effect=RuntimeError("ctor failure"),
    ):
        result = maybe_start_auto_index(tmp_path, auto_index_enabled=True)
    assert result is None


def test_no_event_loop_returns_none(tmp_path: Path) -> None:
    """When called outside an event loop, return None rather than raising."""
    fake_index = MagicMock(is_available=True)
    fake_index.needs_reindex.return_value = True

    def _fake_get_event_loop() -> Any:
        raise RuntimeError("no current event loop")

    with (
        patch("godspeed.context.codebase_index.CodebaseIndex", return_value=fake_index),
        patch("asyncio.get_event_loop", side_effect=_fake_get_event_loop),
    ):
        result = maybe_start_auto_index(tmp_path, auto_index_enabled=True)
    assert result is None
