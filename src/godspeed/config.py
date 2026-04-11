"""Configuration with pydantic-settings. Supports env vars, .env, and YAML config files."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

DEFAULT_GLOBAL_DIR = Path.home() / ".godspeed"
DEFAULT_PROJECT_DIR = Path(".godspeed")


class PermissionSettings(BaseSettings):
    """Permission rules configuration."""

    deny: list[str] = Field(
        default_factory=lambda: [
            # Environment files
            "FileRead(.env)",
            "FileRead(.env.*)",
            "FileRead(.env.local)",
            "FileRead(.env.production)",
            "FileRead(.env.staging)",
            # Keys and certificates
            "FileRead(*.pem)",
            "FileRead(*.key)",
            "FileRead(*.p12)",
            "FileRead(*.pfx)",
            "FileRead(*.jks)",
            # Credential files
            "FileRead(*credentials*)",
            "FileRead(*secret*)",
            # Cloud provider configs
            "FileRead(.aws/*)",
            "FileRead(.gcloud/*)",
            "FileRead(.azure/*)",
            # SSH keys
            "FileRead(.ssh/*)",
            # Docker secrets
            "FileRead(docker-compose*.secret*)",
            # Also block writing to sensitive files
            "FileWrite(.env)",
            "FileWrite(.env.*)",
            "FileWrite(.ssh/*)",
            "FileWrite(.aws/*)",
        ]
    )
    allow: list[str] = Field(
        default_factory=lambda: [
            "Bash(git *)",
            "Bash(ruff *)",
            "Bash(pytest *)",
            "Bash(make *)",
        ]
    )
    ask: list[str] = Field(default_factory=lambda: ["Bash(*)"])

    model_config = SettingsConfigDict(extra="ignore")


class AuditSettings(BaseSettings):
    """Audit trail configuration."""

    enabled: bool = True
    retention_days: int = 30

    model_config = SettingsConfigDict(extra="ignore")


class ContextSettings(BaseSettings):
    """Context management configuration."""

    project_instructions: str = "GODSPEED.md"
    include: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(
        default_factory=lambda: [
            "node_modules/",
            ".venv/",
            "__pycache__/",
            "*.pyc",
            ".git/",
        ]
    )

    model_config = SettingsConfigDict(extra="ignore")


class GodspeedSettings(BaseSettings):
    """Root configuration for Godspeed."""

    # LLM
    model: str = "claude-sonnet-4-20250514"
    fallback_models: list[str] = Field(default_factory=list)

    # Paths
    project_dir: Path = Path(".")
    global_dir: Path = DEFAULT_GLOBAL_DIR

    # Security
    permission_mode: str = "normal"  # "strict" | "normal" | "yolo"

    # Context
    max_context_tokens: int = 100_000
    compaction_threshold: float = 0.8

    # Nested settings
    permissions: PermissionSettings = Field(default_factory=PermissionSettings)
    audit: AuditSettings = Field(default_factory=AuditSettings)
    context: ContextSettings = Field(default_factory=ContextSettings)

    model_config = SettingsConfigDict(
        env_prefix="GODSPEED_",
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
    )

    @field_validator("compaction_threshold")
    @classmethod
    def validate_compaction_threshold(cls, v: float) -> float:
        if not 0.0 < v <= 1.0:
            msg = f"compaction_threshold must be between 0 and 1, got {v}"
            raise ValueError(msg)
        return v

    @field_validator("max_context_tokens")
    @classmethod
    def validate_max_context_tokens(cls, v: int) -> int:
        if v < 1000:
            msg = f"max_context_tokens must be at least 1000, got {v}"
            raise ValueError(msg)
        return v

    @model_validator(mode="before")
    @classmethod
    def load_yaml_configs(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Load and merge YAML config files (global > project). Env vars take precedence."""
        merged: dict[str, Any] = {}

        # Load global config
        global_config = DEFAULT_GLOBAL_DIR / "settings.yaml"
        if global_config.exists():
            with open(global_config) as f:
                global_data = yaml.safe_load(f) or {}
            merged.update(global_data)

        # Load project config (overrides global, except deny rules which merge)
        project_config = DEFAULT_PROJECT_DIR / "settings.yaml"
        if project_config.exists():
            with open(project_config) as f:
                project_data = yaml.safe_load(f) or {}
            _merge_configs(merged, project_data)

        # Env vars / constructor args take final precedence
        merged.update({k: v for k, v in data.items() if v is not None})
        return merged


def _merge_configs(base: dict[str, Any], override: dict[str, Any]) -> None:
    """Merge override into base. Deny rules are additive (project can't weaken global denies)."""
    for key, value in override.items():
        if key == "permissions" and isinstance(value, dict):
            base_perms = base.setdefault("permissions", {})
            # Deny rules are additive — project can only add more denies
            if "deny" in value:
                existing = base_perms.get("deny", [])
                base_perms["deny"] = list(set(existing + value["deny"]))
            # Allow and ask rules: project overrides global
            if "allow" in value:
                base_perms["allow"] = value["allow"]
            if "ask" in value:
                base_perms["ask"] = value["ask"]
        elif isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge_configs(base[key], value)
        else:
            base[key] = value
