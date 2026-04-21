"""Tests for src/godspeed/agent/prompt_profiles.py."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from godspeed.agent.prompt_profiles import (
    PROFILE_PLAN_STYLE,
    PROFILE_PREAMBLES,
    _load_catalog,
    get_catalog_entry,
    plan_style_for,
    preamble_for,
    resolve_profile,
)


@pytest.fixture(autouse=True)
def _clear_catalog_cache() -> None:
    """The catalog is lru_cached; reset between tests."""
    _load_catalog.cache_clear()


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------


def test_known_driver_resolves_to_default() -> None:
    assert resolve_profile("nvidia_nim/moonshotai/kimi-k2.5") == "default"


def test_thinking_driver_resolves_to_thinking() -> None:
    assert resolve_profile("nvidia_nim/moonshotai/kimi-k2-thinking") == "thinking"


def test_small_local_driver_resolves_to_minimal() -> None:
    assert resolve_profile("ollama/qwen3:4b") == "minimal"


def test_unknown_driver_falls_back_to_default() -> None:
    assert resolve_profile("xyz/entirely-made-up-model") == "default"


def test_empty_model_string_falls_back_to_default() -> None:
    assert resolve_profile("") == "default"


# ---------------------------------------------------------------------------
# Preambles / plan hints
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("profile", ["default", "thinking", "minimal"])
def test_each_profile_has_a_preamble(profile: str) -> None:
    text = preamble_for(profile)  # type: ignore[arg-type]
    assert text
    assert isinstance(text, str)


def test_thinking_profile_has_no_plan_hint() -> None:
    # Reasoning models plan internally; explicit plan hurts them.
    assert plan_style_for("thinking") == ""


def test_minimal_profile_has_no_plan_hint() -> None:
    # Tiny models lose focus when asked to plan.
    assert plan_style_for("minimal") == ""


def test_default_profile_has_plan_hint() -> None:
    assert plan_style_for("default")


def test_plan_style_for_unknown_returns_empty() -> None:
    # plan_style_for uses .get() so unknown profile falls through to empty.
    assert plan_style_for("nonexistent") == ""  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Catalog entry accessor
# ---------------------------------------------------------------------------


def test_catalog_entry_returns_full_dict() -> None:
    entry = get_catalog_entry("anthropic/claude-opus-4-7")
    assert entry is not None
    assert entry["context_window"] == 1_000_000
    assert entry["requires_env"] == "ANTHROPIC_API_KEY"
    assert entry["cost_per_mtok_in"] == 15.00


def test_catalog_entry_returns_none_for_unknown() -> None:
    assert get_catalog_entry("xyz/made-up") is None


def test_free_tier_cost_is_zero() -> None:
    entry = get_catalog_entry("nvidia_nim/moonshotai/kimi-k2.5")
    assert entry is not None
    assert entry["cost_per_mtok_in"] == 0.0
    assert entry["cost_per_mtok_out"] == 0.0


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_missing_catalog_file_falls_back_to_default(tmp_path: Path) -> None:
    """If the catalog YAML is missing, everything defaults cleanly."""
    fake_path = tmp_path / "does_not_exist.yaml"
    with patch("godspeed.agent.prompt_profiles._CATALOG_PATH", fake_path):
        _load_catalog.cache_clear()
        assert resolve_profile("nvidia_nim/moonshotai/kimi-k2.5") == "default"
        assert get_catalog_entry("nvidia_nim/moonshotai/kimi-k2.5") is None


def test_malformed_yaml_falls_back_to_default(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("drivers: {not: [a valid\n: mapping", encoding="utf-8")
    with patch("godspeed.agent.prompt_profiles._CATALOG_PATH", bad):
        _load_catalog.cache_clear()
        assert resolve_profile("nvidia_nim/moonshotai/kimi-k2.5") == "default"


def test_driver_with_unknown_profile_falls_back(tmp_path: Path) -> None:
    """Catalog entry referencing a non-existent profile -> default + warning."""
    fake = tmp_path / "catalog.yaml"
    fake.write_text(
        "version: 1\ndrivers:\n  foo/bar:\n    provider: foo\n    prompt_profile: weirdo\n",
        encoding="utf-8",
    )
    with patch("godspeed.agent.prompt_profiles._CATALOG_PATH", fake):
        _load_catalog.cache_clear()
        assert resolve_profile("foo/bar") == "default"


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------


def test_all_profiles_have_preamble_entries() -> None:
    """Every Literal profile name must have a preamble."""
    for name in ("default", "thinking", "minimal"):
        assert name in PROFILE_PREAMBLES


def test_plan_style_keys_are_subset_of_preamble_keys() -> None:
    assert set(PROFILE_PLAN_STYLE.keys()) <= set(PROFILE_PREAMBLES.keys())
