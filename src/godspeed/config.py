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

# Model context windows — used for model-aware compaction prompts.
# Keys are prefixes matched against the model string.
MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    # Frontier models with large context
    "claude-opus": 200_000,
    "claude-sonnet": 200_000,
    "claude-haiku": 200_000,
    "gpt-4o": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-4": 8_192,
    "gpt-3.5": 16_385,
    "gemini-2": 1_000_000,
    "gemini-1.5": 1_000_000,
    "gemini-pro": 32_000,
    # Open-source models (common Ollama defaults)
    "ollama/qwen3": 32_768,
    "ollama/llama3": 8_192,
    "ollama/llama3.1": 128_000,
    "ollama/mistral": 32_768,
    "ollama/codellama": 16_384,
    "ollama/deepseek": 32_768,
    "ollama/gemma": 8_192,
    "ollama/phi": 16_384,
}


def get_model_context_window(model: str) -> int:
    """Get the context window size for a model by prefix matching.

    Returns the matched size or 32_768 as a safe default.
    """
    model_lower = model.lower()
    # Try longest prefix match first for specificity
    best_match = ""
    best_size = 32_768  # safe default
    for prefix, size in MODEL_CONTEXT_WINDOWS.items():
        if model_lower.startswith(prefix.lower()) and len(prefix) > len(best_match):
            best_match = prefix
            best_size = size
    return best_size


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

    # LLM — default to free local Ollama model; override via settings.yaml,
    # GODSPEED_MODEL env var, or `godspeed -m <model>`
    model: str = "ollama/qwen3:4b"
    fallback_models: list[str] = Field(default_factory=list)

    # Paths
    project_dir: Path = Path(".")
    global_dir: Path = DEFAULT_GLOBAL_DIR

    # Security
    permission_mode: str = "normal"  # "strict" | "normal" | "yolo"

    # Context
    max_context_tokens: int = 100_000
    compaction_threshold: float = 0.8

    # Model routing — map task types to specific models
    routing: dict[str, str] = Field(default_factory=dict)

    # MCP servers — each entry is a dict with keys:
    #   name:      str   — unique server identifier (required)
    #   transport: str   — "stdio" (default) or "sse"
    #   command:   str   — executable for stdio transport
    #   args:      list  — CLI args for stdio transport
    #   env:       dict  — extra env vars for stdio subprocess
    #   url:       str   — base URL for sse transport (e.g. "http://localhost:3001")
    #   headers:   dict  — HTTP headers for sse transport (e.g. Authorization)
    mcp_servers: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "MCP server configurations. Each entry supports 'stdio' transport "
            "(command, args, env) or 'sse' transport (url, headers). "
            "Defaults to stdio when transport is omitted."
        ),
    )

    # Hooks — shell commands at lifecycle events
    hooks: list[dict[str, Any]] = Field(default_factory=list)

    # Agent behavior
    parallel_tool_calls: bool = True
    auto_fix_retries: int = 3  # lint-fix retry rounds (0 = one-shot, no auto-fix)
    auto_commit: bool = False
    auto_commit_threshold: int = 5

    # Thinking — extended thinking for Anthropic/Claude models
    thinking_budget: int = 0  # 0 = disabled; >0 = budget_tokens for thinking blocks

    # Cost budget — hard limit on session spend (0 = unlimited)
    max_cost_usd: float = 0.0

    # Architect mode — two-model pipeline (plan then execute)
    architect_model: str = ""  # model for planning phase; empty = use main model

    # Sandboxing
    sandbox: str = "none"  # "none" | "docker"

    # Self-evolution — learn from execution traces to improve prompts/tools
    evolution_enabled: bool = False  # enable with /evolve or config
    evolution_model: str = ""  # model for mutations/judging; empty = auto-detect

    # Memory
    memory_enabled: bool = True

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


def append_allow_rule(pattern: str, project_dir: Path | None = None) -> bool:
    """Append an allow rule to the project or global settings.yaml.

    Reads existing YAML, adds the pattern to permissions.allow, writes back.
    Preserves existing content. Returns True on success.

    Args:
        pattern: Permission pattern to add (e.g. "Shell(git status)").
        project_dir: Project directory for .godspeed/settings.yaml.
            Falls back to global settings if None or project config missing.
    """
    # Determine which settings file to write to
    if project_dir is not None:
        settings_path = project_dir / ".godspeed" / "settings.yaml"
    else:
        settings_path = DEFAULT_GLOBAL_DIR / "settings.yaml"

    try:
        if settings_path.exists():
            with open(settings_path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        else:
            settings_path.parent.mkdir(parents=True, exist_ok=True)
            data = {}

        permissions = data.setdefault("permissions", {})
        allow_list = permissions.setdefault("allow", [])

        if pattern not in allow_list:
            allow_list.append(pattern)

        with open(settings_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)

        logger.info("Appended allow rule '%s' to %s", pattern, settings_path)
        return True
    except OSError as exc:
        logger.warning("Failed to write allow rule: %s", exc)
        return False


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
