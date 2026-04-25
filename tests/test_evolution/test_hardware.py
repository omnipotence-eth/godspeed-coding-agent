"""Tests for hardware-aware model selection."""

from __future__ import annotations

from unittest.mock import patch

import pytest

import godspeed.evolution.hardware as hw_module
from godspeed.evolution.hardware import (
    FALLBACK_MODEL,
    detect_vram_mb,
    is_low_memory,
    recommended_max_eval_cases,
    recommended_num_candidates,
    select_evolution_model,
)


@pytest.fixture(autouse=True)
def _reset_vram_cache():
    """Reset the VRAM cache before and after each test."""
    hw_module._cached_vram = None
    hw_module._cached_vram_checked = False
    yield
    hw_module._cached_vram = None
    hw_module._cached_vram_checked = False


# ---------------------------------------------------------------------------
# Test: detect_vram_mb
# ---------------------------------------------------------------------------


class TestDetectVram:
    def test_nvidia_smi_available(self) -> None:
        with (
            patch("godspeed.evolution.hardware.shutil.which", return_value="/usr/bin/nvidia-smi"),
            patch("godspeed.evolution.hardware.subprocess.run") as mock_run,
        ):
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = "12000\n"
            assert detect_vram_mb() == 12000

    def test_nvidia_smi_not_found(self) -> None:
        with (
            patch("godspeed.evolution.hardware.shutil.which", return_value=None),
            patch("godspeed.evolution.hardware._detect_jetson", return_value=None),
        ):
            assert detect_vram_mb() is None

    def test_nvidia_smi_fails(self) -> None:
        with (
            patch("godspeed.evolution.hardware.shutil.which", return_value="/usr/bin/nvidia-smi"),
            patch("godspeed.evolution.hardware.subprocess.run") as mock_run,
            patch("godspeed.evolution.hardware._detect_jetson", return_value=None),
        ):
            mock_run.return_value.returncode = 1
            assert detect_vram_mb() is None

    def test_nvidia_smi_multi_gpu_uses_first(self) -> None:
        with (
            patch("godspeed.evolution.hardware.shutil.which", return_value="/usr/bin/nvidia-smi"),
            patch("godspeed.evolution.hardware.subprocess.run") as mock_run,
        ):
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = "16000\n8000\n"
            assert detect_vram_mb() == 16000


# ---------------------------------------------------------------------------
# Test: select_evolution_model
# ---------------------------------------------------------------------------


class TestSelectModel:
    def test_explicit_model_respected(self) -> None:
        model = "anthropic/claude-sonnet-4-20250514"
        assert select_evolution_model(model) == model

    def test_high_vram_gets_devstral(self) -> None:
        with patch("godspeed.evolution.hardware._get_cached_vram", return_value=14000):
            assert select_evolution_model() == "ollama/devstral-small-2:24b"

    def test_medium_vram_gets_coder(self) -> None:
        with patch("godspeed.evolution.hardware._get_cached_vram", return_value=9000):
            assert select_evolution_model() == "ollama/qwen2.5-coder:14b"

    def test_low_vram_gets_rnj(self) -> None:
        with patch("godspeed.evolution.hardware._get_cached_vram", return_value=5500):
            assert select_evolution_model() == "ollama/rnj-1:8b"

    def test_very_low_vram_gets_small(self) -> None:
        with patch("godspeed.evolution.hardware._get_cached_vram", return_value=3500):
            assert select_evolution_model() == "ollama/cogito:14b"

    def test_minimal_vram_gets_fallback(self) -> None:
        with patch("godspeed.evolution.hardware._get_cached_vram", return_value=1000):
            # Below 1500 MB threshold — no model fits
            model = select_evolution_model()
            assert model == FALLBACK_MODEL

    def test_no_vram_detected_uses_fallback(self) -> None:
        with patch("godspeed.evolution.hardware._get_cached_vram", return_value=None):
            assert select_evolution_model() == FALLBACK_MODEL

    def test_jetson_orin_nano_8gb(self) -> None:
        """Jetson Orin Nano Super: 8GB shared, ~4.8GB available after 60% factor."""
        with patch("godspeed.evolution.hardware._get_cached_vram", return_value=4800):
            model = select_evolution_model()
            assert model == "ollama/cogito:14b"

    def test_empty_string_triggers_auto(self) -> None:
        with patch("godspeed.evolution.hardware._get_cached_vram", return_value=14000):
            assert select_evolution_model("") == "ollama/devstral-small-2:24b"


# ---------------------------------------------------------------------------
# Test: recommended_num_candidates
# ---------------------------------------------------------------------------


class TestRecommendedCandidates:
    def test_high_vram(self) -> None:
        with patch("godspeed.evolution.hardware._get_cached_vram", return_value=14000):
            assert recommended_num_candidates() == 5

    def test_medium_vram(self) -> None:
        with patch("godspeed.evolution.hardware._get_cached_vram", return_value=7000):
            assert recommended_num_candidates() == 3

    def test_low_vram(self) -> None:
        with patch("godspeed.evolution.hardware._get_cached_vram", return_value=4000):
            assert recommended_num_candidates() == 2

    def test_very_low_vram(self) -> None:
        with patch("godspeed.evolution.hardware._get_cached_vram", return_value=1000):
            assert recommended_num_candidates() == 1

    def test_no_detection(self) -> None:
        with patch("godspeed.evolution.hardware._get_cached_vram", return_value=None):
            assert recommended_num_candidates() == 2

    def test_api_model_no_constraint(self) -> None:
        assert recommended_num_candidates("anthropic/claude-sonnet-4-20250514") == 5


# ---------------------------------------------------------------------------
# Test: recommended_max_eval_cases
# ---------------------------------------------------------------------------


class TestRecommendedEvalCases:
    def test_high_vram(self) -> None:
        with patch("godspeed.evolution.hardware._get_cached_vram", return_value=14000):
            assert recommended_max_eval_cases() == 5

    def test_low_vram(self) -> None:
        with patch("godspeed.evolution.hardware._get_cached_vram", return_value=4000):
            assert recommended_max_eval_cases() == 2

    def test_api_model(self) -> None:
        assert recommended_max_eval_cases("anthropic/claude-sonnet-4-20250514") == 5


# ---------------------------------------------------------------------------
# Test: is_low_memory
# ---------------------------------------------------------------------------


class TestIsLowMemory:
    def test_high_vram_not_low(self) -> None:
        with patch("godspeed.evolution.hardware._get_cached_vram", return_value=14000):
            assert is_low_memory() is False

    def test_low_vram_is_low(self) -> None:
        with patch("godspeed.evolution.hardware._get_cached_vram", return_value=4000):
            assert is_low_memory() is True

    def test_no_detection_assumes_low(self) -> None:
        with patch("godspeed.evolution.hardware._get_cached_vram", return_value=None):
            assert is_low_memory() is True

    def test_boundary_6gb(self) -> None:
        with patch("godspeed.evolution.hardware._get_cached_vram", return_value=6000):
            assert is_low_memory() is False
