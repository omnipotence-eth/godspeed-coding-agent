"""Tests for CorrectionTracker — detection and system prompt injection."""

from __future__ import annotations

from pathlib import Path

import pytest

from godspeed.memory.corrections import CorrectionTracker, is_likely_correction
from godspeed.memory.user_memory import UserMemory


@pytest.fixture
def memory(tmp_path: Path) -> UserMemory:
    mem = UserMemory(db_path=tmp_path / "corrections_test.db")
    yield mem
    mem.close()


@pytest.fixture
def tracker(memory: UserMemory) -> CorrectionTracker:
    return CorrectionTracker(memory)


class TestIsLikelyCorrection:
    """Test the correction detection heuristic."""

    def test_negation_detected(self) -> None:
        assert is_likely_correction("No, don't use print statements") is True

    def test_stop_detected(self) -> None:
        assert is_likely_correction("Stop adding comments to every function") is True

    def test_instead_detected(self) -> None:
        assert is_likely_correction("Use logger instead") is True

    def test_actually_detected(self) -> None:
        assert is_likely_correction("Actually, use snake_case for variables") is True

    def test_prefer_detected(self) -> None:
        assert is_likely_correction("I prefer tabs over spaces") is True

    def test_never_detected(self) -> None:
        assert is_likely_correction("Never use bare except clauses") is True

    def test_always_detected(self) -> None:
        assert is_likely_correction("Always add type hints") is True

    def test_normal_message_not_correction(self) -> None:
        assert is_likely_correction("Write a function to sort a list") is False

    def test_empty_message(self) -> None:
        assert is_likely_correction("") is False

    def test_single_word(self) -> None:
        assert is_likely_correction("hello") is False

    def test_very_long_message(self) -> None:
        long_msg = "no " + "word " * 200
        assert is_likely_correction(long_msg) is False

    def test_please_dont(self) -> None:
        assert is_likely_correction("Please don't mock the database") is True

    def test_wrong_detected(self) -> None:
        assert is_likely_correction("That's wrong, use the other API") is True


class TestCorrectionTracker:
    """Test the CorrectionTracker class."""

    def test_check_detects_and_records(self, tracker: CorrectionTracker) -> None:
        cid = tracker.check_for_correction(
            "No, use logger instead of print",
            last_agent_action="Added print(result)",
        )
        assert cid is not None

    def test_check_ignores_normal_message(self, tracker: CorrectionTracker) -> None:
        cid = tracker.check_for_correction("Write a sorting function")
        assert cid is None

    def test_check_default_action(self, tracker: CorrectionTracker) -> None:
        cid = tracker.check_for_correction("Don't do that")
        assert cid is not None

    def test_get_top_corrections(self, tracker: CorrectionTracker, memory: UserMemory) -> None:
        memory.record_correction("print(x)", "logger.info(x)")
        memory.record_correction("camelCase", "snake_case")
        corrections = tracker.get_top_corrections(n=2)
        assert len(corrections) == 2

    def test_format_empty(self, tracker: CorrectionTracker) -> None:
        result = tracker.format_for_system_prompt()
        assert result == ""

    def test_format_with_corrections(self, tracker: CorrectionTracker, memory: UserMemory) -> None:
        memory.record_correction("used print()", "use logger", context="logging")
        memory.record_correction("camelCase vars", "snake_case vars")
        result = tracker.format_for_system_prompt()
        assert "User corrections" in result
        assert "logger" in result
        assert "snake_case" in result

    def test_format_limits_output(self, tracker: CorrectionTracker, memory: UserMemory) -> None:
        for i in range(20):
            memory.record_correction(f"old_{i}", f"new_{i}")
        result = tracker.format_for_system_prompt(n=3)
        assert result.count("- User said:") == 3
