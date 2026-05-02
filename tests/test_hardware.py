"""Tests for godspeed.evolution.hardware."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from godspeed.evolution.hardware import (
    CANDIDATES_BY_VRAM,
    FALLBACK_MODEL,
    MAX_EVAL_CASES_BY_VRAM,
    MODEL_TIERS,
    MachineSpecs,
    _detect_gpu_name,
    _detect_jetson,
    _detect_nvidia_smi,
    _get_cached_vram,
    is_low_memory,
    recommend_models_for_machine,
    recommended_max_eval_cases,
    recommended_num_candidates,
    scan_machine,
    select_evolution_model,
)


class TestConstants:
    def test_fallback_model(self):
        assert FALLBACK_MODEL == "ollama/rnj-1:8b"

    def test_model_tiers_not_empty(self):
        assert len(MODEL_TIERS) > 0
        # Each tier is (min_vram_mb, model, desc)
        assert isinstance(MODEL_TIERS[0], tuple)
        assert len(MODEL_TIERS[0]) == 3

    def test_candidates_by_vram(self):
        assert len(CANDIDATES_BY_VRAM) > 0
        # Sorted descending by VRAM
        for i in range(len(CANDIDATES_BY_VRAM) - 1):
            assert CANDIDATES_BY_VRAM[i][0] >= CANDIDATES_BY_VRAM[i + 1][0]

    def test_max_eval_cases_by_vram(self):
        assert len(MAX_EVAL_CASES_BY_VRAM) > 0


class TestVramDetection:
    def test_get_cached_vram_first_call(self):
        with patch("godspeed.evolution.hardware._cached_vram_checked", False):
            with patch("godspeed.evolution.hardware.detect_vram_mb", return_value=8000):
                vram = _get_cached_vram()
                assert vram == 8000

    def test_get_cached_vram_cached(self):
        with patch("godspeed.evolution.hardware._cached_vram", 12000):
            with patch("godspeed.evolution.hardware._cached_vram_checked", True):
                vram = _get_cached_vram()
                assert vram == 12000

    def test_detect_nvidia_smi_success(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "8192\n"
        with patch("subprocess.run", return_value=mock_result):
            vram = _detect_nvidia_smi()
            assert vram == 8192

    def test_detect_nvidia_smi_not_found(self):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            vram = _detect_nvidia_smi()
            assert vram is None

    def test_detect_nvidia_smi_failure(self):
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        with patch("subprocess.run", return_value=mock_result):
            vram = _detect_nvidia_smi()
            assert vram is None

    def test_detect_jetson_not_jetson(self):
        # _detect_jetson checks Path("/sys/class/thermal/thermal_zone0/temp")
        # Patch the specific path check by mocking Path.exists
        original_exists = Path.exists

        def mock_exists(self):
            if "thermal" in str(self).lower() or "jetson" in str(self).lower():
                return False
            return original_exists(self)

        with patch("pathlib.Path.exists", mock_exists):
            vram = _detect_jetson()
            assert vram is None

    def test_detect_gpu_name_nvidia(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="NVIDIA GeForce RTX 5070 Ti\n")
            name = _detect_gpu_name()
            assert "NVIDIA" in name or "RTX" in name

    def test_detect_gpu_name_not_found(self):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            name = _detect_gpu_name()
            assert name == ""


class TestSelectEvolutionModel:
    def test_explicit_config(self):
        model = select_evolution_model(configured_model="ollama/qwen2.5-coder:14b")
        assert model == "ollama/qwen2.5-coder:14b"

    def test_auto_vram_sufficient(self):
        with patch("godspeed.evolution.hardware._get_cached_vram", return_value=12000):
            model = select_evolution_model()
            # Should pick the largest model that fits
            assert "24b" in model or "14b" in model

    def test_auto_vram_low(self):
        with patch("godspeed.evolution.hardware._get_cached_vram", return_value=3000):
            model = select_evolution_model()
            # 3000 MB matches cogito:14b (min_vram=3000)
            assert "14b" in model or "cogito" in model

    def test_auto_vram_none(self):
        with patch("godspeed.evolution.hardware._get_cached_vram", return_value=None):
            model = select_evolution_model()
            assert model == FALLBACK_MODEL


class TestRecommendedNumCandidates:
    def test_api_model(self):
        # Non-ollama models don't have VRAM constraint
        num = recommended_num_candidates(configured_model="anthropic/claude-sonnet-4-6")
        assert num == 5

    def test_vram_sufficient(self):
        with patch("godspeed.evolution.hardware._get_cached_vram", return_value=12000):
            num = recommended_num_candidates()
            assert num >= 3

    def test_vram_low(self):
        with patch("godspeed.evolution.hardware._get_cached_vram", return_value=3000):
            num = recommended_num_candidates()
            assert num >= 1

    def test_vram_none(self):
        with patch("godspeed.evolution.hardware._get_cached_vram", return_value=None):
            num = recommended_num_candidates()
            assert num >= 1


class TestRecommendedMaxEvalCases:
    def test_vram_sufficient(self):
        with patch("godspeed.evolution.hardware._get_cached_vram", return_value=12000):
            num = recommended_max_eval_cases()
            assert num >= 3

    def test_vram_low(self):
        with patch("godspeed.evolution.hardware._get_cached_vram", return_value=3000):
            num = recommended_max_eval_cases()
            assert num >= 1


class TestIsLowMemory:
    def test_low_memory_vram(self):
        with patch("godspeed.evolution.hardware._get_cached_vram", return_value=2000):
            assert is_low_memory() is True

    def test_normal_memory_vram(self):
        with patch("godspeed.evolution.hardware._get_cached_vram", return_value=8000):
            assert is_low_memory() is False

    def test_no_vram_info(self):
        with patch("godspeed.evolution.hardware._get_cached_vram", return_value=None):
            # No VRAM info -> assume constrained (conservative)
            assert is_low_memory() is True


class TestMachineSpecs:
    def test_defaults(self):
        specs = MachineSpecs(platform="windows", vram_mb=0, ram_gb=0.0, cpu_cores=1, gpu_name="")
        assert specs.platform == "windows"
        assert specs.ram_gb == 0.0
        assert specs.vram_mb == 0
        assert specs.gpu_name == ""
        assert specs.cpu_cores == 1

    def test_custom(self):
        specs = MachineSpecs(
            platform="windows", vram_mb=16384, ram_gb=96.0, cpu_cores=16, gpu_name="RTX 5070 Ti"
        )
        assert specs.platform == "windows"
        assert specs.cpu_cores == 16
        assert specs.ram_gb == 96.0
        assert specs.vram_mb == 16384
        assert specs.gpu_name == "RTX 5070 Ti"


class TestScanMachine:
    def test_scan(self):
        with patch("godspeed.evolution.hardware._detect_gpu_name", return_value="RTX 5070 Ti"):
            with patch("godspeed.evolution.hardware._get_cached_vram", return_value=16384):
                with patch("psutil.virtual_memory") as mock_mem:
                    mock_mem.return_value = MagicMock(total=96 * 1024**3)
                    specs = scan_machine()
                    assert specs.gpu_name == "RTX 5070 Ti"
                    assert specs.vram_mb == 16384
                    assert specs.ram_gb >= 90


class TestRecommendModelsForMachine:
    def test_with_specs(self):
        specs = MachineSpecs(
            platform="windows", vram_mb=16384, ram_gb=96.0, cpu_cores=16, gpu_name="RTX 5070 Ti"
        )
        recs = recommend_models_for_machine(specs=specs)
        assert isinstance(recs, dict)
        assert "fast" in recs or "balanced" in recs

    def test_without_specs(self):
        with patch("godspeed.evolution.hardware.scan_machine") as mock_scan:
            mock_scan.return_value = MachineSpecs(
                platform="windows", vram_mb=8000, ram_gb=32.0, cpu_cores=8, gpu_name="RTX 5070 Ti"
            )
            recs = recommend_models_for_machine()
            assert isinstance(recs, dict)

    def test_low_vram(self):
        specs = MachineSpecs(
            platform="windows", vram_mb=2000, ram_gb=16.0, cpu_cores=4, gpu_name=""
        )
        recs = recommend_models_for_machine(specs=specs)
        assert isinstance(recs, dict)
