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


# ---------------------------------------------------------------------------
# Test: detect_vram_mb Jetson fallback path (line 89-91)
# ---------------------------------------------------------------------------


class TestDetectVramJetson:
    def test_jetson_path_after_nvidia_smi_fails(self) -> None:
        with (
            patch("godspeed.evolution.hardware.shutil.which", return_value=None),
            patch("godspeed.evolution.hardware._detect_jetson", return_value=4800),
        ):
            assert detect_vram_mb() == 4800


# ---------------------------------------------------------------------------
# Test: _detect_nvidia_smi with float value (line 119)
# ---------------------------------------------------------------------------


class TestDetectNvidiaSmiDecimal:
    def test_parses_float_output_as_int(self) -> None:
        with (
            patch("godspeed.evolution.hardware.shutil.which", return_value="/usr/bin/nvidia-smi"),
            patch("godspeed.evolution.hardware.subprocess.run") as mock_run,
        ):
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = "8192.7\n"
            assert detect_vram_mb() == 8192

    def test_value_error_on_non_numeric(self) -> None:
        with (
            patch("godspeed.evolution.hardware.shutil.which", return_value="/usr/bin/nvidia-smi"),
            patch("godspeed.evolution.hardware.subprocess.run") as mock_run,
            patch("godspeed.evolution.hardware._detect_jetson", return_value=None),
        ):
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = "not_a_number\n"
            assert detect_vram_mb() is None


# ---------------------------------------------------------------------------
# Test: scan_machine with cpu_count returning None
# ---------------------------------------------------------------------------


class TestScanMachineCpuFallback:
    def test_cpu_count_none_falls_back_to_one(self) -> None:
        with (
            patch("godspeed.evolution.hardware._get_cached_vram", return_value=8000),
            patch("godspeed.evolution.hardware._detect_gpu_name", return_value="Test GPU"),
            patch("psutil.virtual_memory") as mock_mem,
        ):
            from unittest.mock import MagicMock
            mock_mem.return_value = MagicMock(total=16 * 1024**3)
            with patch("os.cpu_count", return_value=None):
                specs = hw_module.scan_machine()
                assert specs.cpu_cores == 1
                assert specs.platform is not None
                assert specs.ram_gb >= 15


# ---------------------------------------------------------------------------
# Test: recommend_models_for_machine — CPU mode budget calculation
# ---------------------------------------------------------------------------


class TestRecommendModelsCpuMode:
    def test_cpu_mode_budget_calculation(self) -> None:
        from godspeed.evolution.hardware import MachineSpecs, recommend_models_for_machine

        specs = MachineSpecs(
            platform="linux", vram_mb=None, ram_gb=8.0, cpu_cores=4, gpu_name=""
        )
        recs = recommend_models_for_machine(specs)
        # CPU budget = int(8.0 * 1024 * 0.6) = 4915 MB → should get fallback (1500)
        # and maybe fast_alt (3000)
        assert recs["fast"] is not None
        assert recs["cloud"] is not None
        assert recs["frontier"] is not None

    def test_zero_vram_cpu_mode_budget(self) -> None:
        from godspeed.evolution.hardware import MachineSpecs, recommend_models_for_machine

        specs = MachineSpecs(
            platform="linux", vram_mb=0, ram_gb=32.0, cpu_cores=8, gpu_name=""
        )
        recs = recommend_models_for_machine(specs)
        assert recs["fast"] is not None
        assert "balanced" in recs

    def test_cpu_mode_low_ram_only_fallback(self) -> None:
        from godspeed.evolution.hardware import MachineSpecs, recommend_models_for_machine

        specs = MachineSpecs(
            platform="linux", vram_mb=0, ram_gb=1.0, cpu_cores=2, gpu_name=""
        )
        recs = recommend_models_for_machine(specs)
        # Budget = int(1.0 * 1024 * 0.6) = 614 MB — below 1500 fallback minimum
        # So only the initial selected_fast (ollama/qwen2.5:1.5b) persists
        assert recs["fast"] is not None
        assert recs["fast"] == "ollama/qwen2.5:1.5b"


# ---------------------------------------------------------------------------
# Test: format_machine_report — no specs calls scan_machine
# ---------------------------------------------------------------------------


class TestFormatMachineReportEdgeCases:
    def test_format_with_none_vram_and_empty_gpu(self) -> None:
        from godspeed.evolution.hardware import MachineSpecs, format_machine_report

        specs = MachineSpecs(
            platform="linux", vram_mb=None, ram_gb=4.0, cpu_cores=2, gpu_name=""
        )
        report = format_machine_report(specs)
        assert "GODSPEED MACHINE SCAN" in report
        assert "CPU-only mode" in report

    def test_format_with_zero_vram_shows_0_gb(self) -> None:
        from godspeed.evolution.hardware import MachineSpecs, format_machine_report

        specs = MachineSpecs(
            platform="windows", vram_mb=0, ram_gb=16.0, cpu_cores=8, gpu_name="NVIDIA RTX"
        )
        report = format_machine_report(specs)
        assert "GODSPEED MACHINE SCAN" in report

    def test_format_with_none_specs_calls_scan(self) -> None:
        from godspeed.evolution.hardware import format_machine_report

        with patch("godspeed.evolution.hardware.scan_machine") as mock_scan:
            from godspeed.evolution.hardware import MachineSpecs
            mock_scan.return_value = MachineSpecs(
                platform="darwin", vram_mb=None, ram_gb=32.0, cpu_cores=10, gpu_name="M3"
            )
            report = format_machine_report()
            assert "GODSPEED MACHINE SCAN" in report
            assert "darwin" in report
            mock_scan.assert_called_once()

    def test_format_report_unix_platform(self) -> None:
        from godspeed.evolution.hardware import MachineSpecs, format_machine_report

        specs = MachineSpecs(
            platform="linux", vram_mb=12000, ram_gb=64.0, cpu_cores=16, gpu_name="A100"
        )
        report = format_machine_report(specs)
        assert "RECOMMENDED MODELS" in report
        assert "--preset" in report
        assert "✓" in report
