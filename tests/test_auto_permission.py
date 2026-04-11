"""Tests for auto-permission suggestion flow."""

from __future__ import annotations

from pathlib import Path

import yaml

from godspeed.config import append_allow_rule
from godspeed.security.approval_tracker import ApprovalTracker


class TestAppendAllowRule:
    """Test append_allow_rule() config persistence."""

    def test_creates_new_file(self, tmp_path: Path) -> None:
        project = tmp_path / "project"
        project.mkdir()
        result = append_allow_rule("Shell(git status)", project_dir=project)
        assert result is True

        settings_path = project / ".godspeed" / "settings.yaml"
        assert settings_path.exists()
        data = yaml.safe_load(settings_path.read_text(encoding="utf-8"))
        assert "Shell(git status)" in data["permissions"]["allow"]

    def test_appends_to_existing(self, tmp_path: Path) -> None:
        project = tmp_path / "project"
        godspeed_dir = project / ".godspeed"
        godspeed_dir.mkdir(parents=True)
        settings_path = godspeed_dir / "settings.yaml"
        settings_path.write_text(
            yaml.safe_dump(
                {"permissions": {"allow": ["Shell(ruff *)"]}},
                default_flow_style=False,
            ),
            encoding="utf-8",
        )

        result = append_allow_rule("Shell(git status)", project_dir=project)
        assert result is True

        data = yaml.safe_load(settings_path.read_text(encoding="utf-8"))
        assert "Shell(ruff *)" in data["permissions"]["allow"]
        assert "Shell(git status)" in data["permissions"]["allow"]

    def test_no_duplicates(self, tmp_path: Path) -> None:
        project = tmp_path / "project"
        godspeed_dir = project / ".godspeed"
        godspeed_dir.mkdir(parents=True)
        settings_path = godspeed_dir / "settings.yaml"
        settings_path.write_text(
            yaml.safe_dump(
                {"permissions": {"allow": ["Shell(git status)"]}},
                default_flow_style=False,
            ),
            encoding="utf-8",
        )

        result = append_allow_rule("Shell(git status)", project_dir=project)
        assert result is True

        data = yaml.safe_load(settings_path.read_text(encoding="utf-8"))
        assert data["permissions"]["allow"].count("Shell(git status)") == 1

    def test_preserves_other_settings(self, tmp_path: Path) -> None:
        project = tmp_path / "project"
        godspeed_dir = project / ".godspeed"
        godspeed_dir.mkdir(parents=True)
        settings_path = godspeed_dir / "settings.yaml"
        settings_path.write_text(
            yaml.safe_dump(
                {"model": "gpt-4o", "permissions": {"deny": ["rm -rf *"]}},
                default_flow_style=False,
            ),
            encoding="utf-8",
        )

        append_allow_rule("Shell(git status)", project_dir=project)

        data = yaml.safe_load(settings_path.read_text(encoding="utf-8"))
        assert data["model"] == "gpt-4o"
        assert "rm -rf *" in data["permissions"]["deny"]
        assert "Shell(git status)" in data["permissions"]["allow"]


class TestAutoPermissionFlow:
    """Test the full approval tracking → suggestion flow."""

    def test_three_approvals_triggers_suggestion(self) -> None:
        tracker = ApprovalTracker(threshold=3)
        pattern = "Shell(git status)"

        for _ in range(3):
            tracker.record_approval(pattern)

        assert tracker.should_suggest(pattern)

    def test_suggestion_only_fires_once(self) -> None:
        tracker = ApprovalTracker(threshold=2)
        pattern = "Shell(git status)"

        tracker.record_approval(pattern)
        tracker.record_approval(pattern)
        assert tracker.should_suggest(pattern)

        # Additional approvals don't re-trigger
        tracker.record_approval(pattern)
        assert not tracker.should_suggest(pattern)

    def test_different_patterns_independent(self) -> None:
        tracker = ApprovalTracker(threshold=2)

        tracker.record_approval("Shell(git status)")
        tracker.record_approval("Shell(npm test)")
        tracker.record_approval("Shell(git status)")

        assert tracker.should_suggest("Shell(git status)")
        assert not tracker.should_suggest("Shell(npm test)")
