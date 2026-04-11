"""Tests for configuration system."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from godspeed.config import GodspeedSettings, PermissionSettings, _merge_configs


class TestGodspeedSettings:
    """Test root configuration."""

    def test_defaults(self, settings: GodspeedSettings) -> None:
        assert settings.model == "ollama/qwen3:4b"
        assert settings.permission_mode == "normal"
        assert settings.max_context_tokens == 100_000
        assert settings.compaction_threshold == 0.8
        assert settings.audit.enabled is True

    def test_env_override(self, tmp_project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("godspeed.config.DEFAULT_GLOBAL_DIR", tmp_project / ".gs-global")
        monkeypatch.setattr("godspeed.config.DEFAULT_PROJECT_DIR", tmp_project / ".godspeed")
        monkeypatch.setenv("GODSPEED_MODEL", "gpt-4o")
        monkeypatch.setenv("GODSPEED_PERMISSION_MODE", "strict")
        s = GodspeedSettings(project_dir=tmp_project)
        assert s.model == "gpt-4o"
        assert s.permission_mode == "strict"

    def test_yaml_global_config(self, tmp_project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        global_dir = tmp_project / ".gs-global"
        global_dir.mkdir()
        config = {"model": "ollama/llama3", "permission_mode": "strict"}
        (global_dir / "settings.yaml").write_text(yaml.dump(config))
        monkeypatch.setattr("godspeed.config.DEFAULT_GLOBAL_DIR", global_dir)
        monkeypatch.setattr("godspeed.config.DEFAULT_PROJECT_DIR", tmp_project / ".godspeed")
        s = GodspeedSettings(project_dir=tmp_project)
        assert s.model == "ollama/llama3"
        assert s.permission_mode == "strict"

    def test_yaml_project_overrides_global(
        self, tmp_project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        global_dir = tmp_project / ".gs-global"
        global_dir.mkdir()
        (global_dir / "settings.yaml").write_text(yaml.dump({"model": "gpt-4o"}))
        project_dir = tmp_project / ".godspeed"
        project_dir.mkdir(exist_ok=True)
        (project_dir / "settings.yaml").write_text(yaml.dump({"model": "claude-sonnet-4-20250514"}))
        monkeypatch.setattr("godspeed.config.DEFAULT_GLOBAL_DIR", global_dir)
        monkeypatch.setattr("godspeed.config.DEFAULT_PROJECT_DIR", project_dir)
        s = GodspeedSettings(project_dir=tmp_project)
        assert s.model == "claude-sonnet-4-20250514"


class TestPermissionSettings:
    """Test permission defaults."""

    def test_default_deny_rules(self) -> None:
        p = PermissionSettings()
        assert "FileRead(.env)" in p.deny
        assert "FileRead(*.pem)" in p.deny

    def test_default_allow_rules(self) -> None:
        p = PermissionSettings()
        assert "Bash(git *)" in p.allow
        assert "Bash(pytest *)" in p.allow

    def test_default_ask_rules(self) -> None:
        p = PermissionSettings()
        assert "Bash(*)" in p.ask


class TestMergeConfigs:
    """Test config merging logic."""

    def test_deny_rules_are_additive(self) -> None:
        base = {"permissions": {"deny": ["FileRead(.env)"]}}
        override = {"permissions": {"deny": ["FileRead(*.key)"]}}
        _merge_configs(base, override)
        assert "FileRead(.env)" in base["permissions"]["deny"]
        assert "FileRead(*.key)" in base["permissions"]["deny"]

    def test_project_cannot_remove_global_deny(self) -> None:
        base = {"permissions": {"deny": ["FileRead(.env)", "Bash(rm -rf *)"]}}
        override = {"permissions": {"deny": []}}
        _merge_configs(base, override)
        # Global denies are preserved
        assert "FileRead(.env)" in base["permissions"]["deny"]
        assert "Bash(rm -rf *)" in base["permissions"]["deny"]

    def test_allow_rules_override(self) -> None:
        base = {"permissions": {"allow": ["Bash(git *)"]}}
        override = {"permissions": {"allow": ["Bash(npm *)"]}}
        _merge_configs(base, override)
        assert base["permissions"]["allow"] == ["Bash(npm *)"]

    def test_non_permission_keys_override(self) -> None:
        base = {"model": "gpt-4o"}
        override = {"model": "claude-sonnet-4-20250514"}
        _merge_configs(base, override)
        assert base["model"] == "claude-sonnet-4-20250514"
