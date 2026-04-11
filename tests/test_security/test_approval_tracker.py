"""Tests for the auto-permission approval tracker."""

from __future__ import annotations

import threading

from godspeed.security.approval_tracker import ApprovalTracker


class TestApprovalTracker:
    """Test ApprovalTracker."""

    def test_initial_count_is_zero(self) -> None:
        tracker = ApprovalTracker()
        assert tracker.get_count("Shell(git status)") == 0

    def test_record_increments_count(self) -> None:
        tracker = ApprovalTracker()
        tracker.record_approval("Shell(git status)")
        assert tracker.get_count("Shell(git status)") == 1
        tracker.record_approval("Shell(git status)")
        assert tracker.get_count("Shell(git status)") == 2

    def test_separate_patterns_tracked_independently(self) -> None:
        tracker = ApprovalTracker()
        tracker.record_approval("Shell(git status)")
        tracker.record_approval("Shell(npm test)")
        assert tracker.get_count("Shell(git status)") == 1
        assert tracker.get_count("Shell(npm test)") == 1

    def test_should_suggest_at_threshold(self) -> None:
        tracker = ApprovalTracker(threshold=3)
        pattern = "Shell(git status)"
        for _ in range(2):
            tracker.record_approval(pattern)
            assert not tracker.should_suggest(pattern)

        tracker.record_approval(pattern)
        assert tracker.should_suggest(pattern)

    def test_should_suggest_only_once(self) -> None:
        tracker = ApprovalTracker(threshold=2)
        pattern = "Shell(git status)"
        tracker.record_approval(pattern)
        tracker.record_approval(pattern)

        assert tracker.should_suggest(pattern)
        # Second call returns False even with more approvals
        tracker.record_approval(pattern)
        assert not tracker.should_suggest(pattern)

    def test_custom_threshold_override(self) -> None:
        tracker = ApprovalTracker(threshold=5)
        pattern = "Shell(git status)"
        for _ in range(3):
            tracker.record_approval(pattern)

        # Default threshold is 5, but we override to 3
        assert tracker.should_suggest(pattern, threshold=3)

    def test_below_threshold_returns_false(self) -> None:
        tracker = ApprovalTracker(threshold=3)
        tracker.record_approval("Shell(git status)")
        assert not tracker.should_suggest("Shell(git status)")

    def test_reset_clears_all(self) -> None:
        tracker = ApprovalTracker(threshold=2)
        pattern = "Shell(git status)"
        tracker.record_approval(pattern)
        tracker.record_approval(pattern)
        tracker.should_suggest(pattern)

        tracker.reset()
        assert tracker.get_count(pattern) == 0
        # After reset, suggestion can trigger again
        tracker.record_approval(pattern)
        tracker.record_approval(pattern)
        assert tracker.should_suggest(pattern)

    def test_thread_safety(self) -> None:
        tracker = ApprovalTracker(threshold=100)
        pattern = "Shell(git status)"

        def record_many() -> None:
            for _ in range(100):
                tracker.record_approval(pattern)

        threads = [threading.Thread(target=record_many) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert tracker.get_count(pattern) == 1000

    def test_should_suggest_with_zero_threshold(self) -> None:
        tracker = ApprovalTracker(threshold=0)
        # Even with 0 threshold, should suggest after first check
        assert tracker.should_suggest("Shell(git status)")
