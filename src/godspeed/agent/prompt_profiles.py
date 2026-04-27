"""Per-driver prompt profile registry.

Different model families respond to different system-prompt shapes:

- ``default`` — general instruction-tuned chat models (Kimi K2.5/K2.6,
  GPT-OSS, Qwen Coder, Claude Sonnet). Standard guidance is well-tolerated.
- ``thinking`` — extended-thinking / reasoning models (Qwen3-Next Thinking,
  DeepSeek R1, Kimi-K2-thinking, o1-style). Overspecification hurts
  these; they do best with a very terse goal + stop criterion and free
  thinking budget.
- ``minimal`` — tiny and structure-rigid models (Ollama qwen3:4b,
  gemma3:1b). Long prompts drown signal; use a short directive-only form.

Selection:

    profile = resolve_profile(model="nvidia_nim/moonshotai/kimi-k2.5")
    #  -> "default" (from driver_catalog.yaml)
    #  -> or "default" fallback if not in catalog

The catalog is the source of truth. This module provides the lookup
plus the prompt additions each profile applies.

Scope: this module does NOT build the full system prompt — that lives
in ``godspeed.agent.system_prompt.build_system_prompt``. Profiles add
small directional tweaks on top (tone of the preamble, whether to spell
out a PLAN/EXECUTE structure, etc.). The main system prompt is profile-
agnostic.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

ProfileName = Literal["default", "thinking", "minimal"]

_CATALOG_PATH = Path(__file__).resolve().parent.parent / "llm" / "driver_catalog.yaml"


# ---------------------------------------------------------------------------
# Profile definitions
# ---------------------------------------------------------------------------


PROFILE_PREAMBLES: dict[ProfileName, str] = {
    "default": (
        "You are a careful senior software engineer. Read before writing, "
        "test before declaring done, and prefer minimal targeted fixes."
    ),
    "thinking": (
        "You are a reasoning model. Think thoroughly before acting, but "
        "keep visible output focused on the fix. Edit the smallest possible "
        "set of files."
    ),
    "minimal": ("Fix the bug. Read the failing test. Edit the source file. Stop."),
}


PROFILE_PLAN_STYLE: dict[ProfileName, str] = {
    # Phrase used to ask the agent to plan before acting; empty = skip.
    "default": (
        "Before editing, briefly state (1) what file + function to modify "
        "and (2) what the minimal fix is."
    ),
    "thinking": "",  # reasoning models plan internally; explicit plan adds noise
    "minimal": "",  # tiny models lose focus if asked to plan
}


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _load_catalog() -> dict[str, dict]:
    """Load driver_catalog.yaml; cached for the process lifetime.

    Returns a flat ``{model_string: entry_dict}`` mapping. If the catalog
    file is missing or malformed, returns an empty dict (falls back to
    ``default`` profile for all models).
    """
    try:
        import yaml
    except ImportError:
        logger.warning(
            "pyyaml not installed; driver catalog unavailable, all models will use default profile"
        )
        return {}

    if not _CATALOG_PATH.is_file():
        logger.debug("driver_catalog.yaml not found at %s", _CATALOG_PATH)
        return {}

    try:
        data = yaml.safe_load(_CATALOG_PATH.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        logger.warning("driver_catalog.yaml is invalid: %s", exc)
        return {}

    drivers = (data or {}).get("drivers") or {}
    if not isinstance(drivers, dict):
        logger.warning("driver_catalog.yaml 'drivers' is not a mapping; ignoring")
        return {}
    return drivers


def resolve_profile(model: str) -> ProfileName:
    """Return the prompt profile name for a given LiteLLM model string.

    Falls back to ``default`` if the model is not in the catalog. A
    fuzzy substring match is NOT used — if you want a new driver on a
    non-default profile, add it to ``driver_catalog.yaml`` explicitly.
    """
    drivers = _load_catalog()
    entry = drivers.get(model)
    if not entry:
        return "default"
    raw = entry.get("prompt_profile", "default")
    if raw not in PROFILE_PREAMBLES:
        logger.warning(
            "driver_catalog.yaml: model %s has unknown prompt_profile %r; using default",
            model,
            raw,
        )
        return "default"
    return raw


def preamble_for(profile: ProfileName) -> str:
    """Return the short system-prompt preamble for this profile."""
    return PROFILE_PREAMBLES[profile]


def plan_style_for(profile: ProfileName) -> str:
    """Return the plan-style hint for this profile, or empty if none."""
    return PROFILE_PLAN_STYLE.get(profile, "")


def get_catalog_entry(model: str) -> dict | None:
    """Return the full catalog entry for a model, or None if unknown.

    Callers that want the context window, cost, or known_ceilings should
    use this rather than re-parsing the YAML.
    """
    return _load_catalog().get(model)
