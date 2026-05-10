"""Tests for godspeed.evolution.hardware."""

from __future__ import annotations

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
        with patch("shutil.which", return_value="nvidia-smi"):
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
        # _detect_jetson reads /proc/meminfo directly with open()
        # Mock open to raise FileNotFoundError to simulate non-Jetson
        with patch("builtins.open", side_effect=FileNotFoundError):
            vram = _detect_jetson()
            assert vram is None

    def test_detect_gpu_name_nvidia(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "NVIDIA GeForce RTX 5070 Ti\n"
        with patch("shutil.which", return_value="nvidia-smi"):
            with patch("subprocess.run", return_value=mock_result):
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

    def test_no_vram_cpu_mode(self):
        specs = MachineSpecs(
            platform="linux", vram_mb=None, ram_gb=32.0, cpu_cores=8, gpu_name=""
        )
        recs = recommend_models_for_machine(specs=specs)
        assert isinstance(recs, dict)
        assert recs["fast"] is not None
        assert recs["cloud"] is not None
        assert recs["frontier"] is not None

    def test_zero_vram_cpu_mode(self):
        specs = MachineSpecs(
            platform="linux", vram_mb=0, ram_gb=16.0, cpu_cores=4, gpu_name=""
        )
        recs = recommend_models_for_machine(specs=specs)
        assert isinstance(recs, dict)
        assert recs["fast"] is not None

    def test_all_tier_keys_present(self):
        specs = MachineSpecs(
            platform="windows", vram_mb=16384, ram_gb=96.0, cpu_cores=16, gpu_name="RTX 5070 Ti"
        )
        recs = recommend_models_for_machine(specs=specs)
        for tier in ("fast", "balanced", "quality", "cloud", "frontier"):
            assert tier in recs

    def test_fast_tier_never_none(self):
        specs = MachineSpecs(
            platform="windows", vram_mb=500, ram_gb=4.0, cpu_cores=2, gpu_name=""
        )
        recs = recommend_models_for_machine(specs=specs)
        assert recs["fast"] is not None

    def test_no_vram_detected_quality_is_none(self):
        specs = MachineSpecs(
            platform="linux", vram_mb=None, ram_gb=8.0, cpu_cores=4, gpu_name=""
        )
        recs = recommend_models_for_machine(specs=specs)
        assert recs.get("quality") is None

    def test_darwin_platform(self):
        specs = MachineSpecs(
            platform="darwin", vram_mb=None, ram_gb=16.0, cpu_cores=8, gpu_name=""
        )
        recs = recommend_models_for_machine(specs=specs)
        assert isinstance(recs, dict)
        assert recs["fast"] is not None


class TestFormatMachineReport:
    def test_with_specs(self):
        from godspeed.evolution.hardware import format_machine_report

        specs = MachineSpecs(
            platform="windows", vram_mb=16384, ram_gb=96.0, cpu_cores=16, gpu_name="RTX 5070 Ti"
        )
        report = format_machine_report(specs)
        assert "GODSPEED MACHINE SCAN" in report
        assert "windows" in report
        assert "RTX 5070 Ti" in report
        assert "96.0 GB" in report
        assert "16.0 GB free" in report

    def test_without_gpu(self):
        from godspeed.evolution.hardware import format_machine_report

        specs = MachineSpecs(
            platform="linux", vram_mb=None, ram_gb=8.0, cpu_cores=4, gpu_name=""
        )
        report = format_machine_report(specs)
        assert "CPU-only mode" in report

    def test_without_specs_calls_scan(self):
        from godspeed.evolution.hardware import format_machine_report

        with patch("godspeed.evolution.hardware.scan_machine") as mock_scan:
            mock_scan.return_value = MachineSpecs(
                platform="linux", vram_mb=8000, ram_gb=32.0, cpu_cores=8, gpu_name=""
            )
            report = format_machine_report()
            assert "GODSPEED MACHINE SCAN" in report
            mock_scan.assert_called_once()

    def test_quality_insufficient_vram_shows_cross(self):
        from godspeed.evolution.hardware import format_machine_report

        specs = MachineSpecs(
            platform="windows", vram_mb=2000, ram_gb=8.0, cpu_cores=4, gpu_name=""
        )
        report = format_machine_report(specs)
        assert "insufficient VRAM" in report or "\u2717" in report

    def test_all_tiers_present_in_report(self):
        from godspeed.evolution.hardware import format_machine_report

        specs = MachineSpecs(
            platform="windows", vram_mb=16384, ram_gb=96.0, cpu_cores=16, gpu_name="RTX 5070 Ti"
        )
        report = format_machine_report(specs)
        assert "Fast" in report
        assert "Balanced" in report
        assert "Quality" in report
        assert "Cloud" in report
        assert "Frontier" in report

    def test_preset_switch_hint_present(self):
        from godspeed.evolution.hardware import format_machine_report

        specs = MachineSpecs(
            platform="windows", vram_mb=12000, ram_gb=32.0, cpu_cores=8, gpu_name="RTX 5070 Ti"
        )
        report = format_machine_report(specs)
        assert "--preset" in report


class TestScanMachineEdgeCases:
    def test_psutil_import_error_fallback(self):
        with patch("godspeed.evolution.hardware._get_cached_vram", return_value=8000):
            with patch("godspeed.evolution.hardware._detect_gpu_name", return_value=""):
                with patch(
                    "builtins.__import__",
                    side_effect=ImportError("no psutil"),
                ):
                    specs = scan_machine()
                    assert specs.ram_gb == 0.0
                    assert specs.cpu_cores >= 1

    def test_scan_machine_without_gpu(self):
        with patch("godspeed.evolution.hardware._get_cached_vram", return_value=None):
            with patch("godspeed.evolution.hardware._detect_gpu_name", return_value=""):
                with patch("psutil.virtual_memory") as mock_mem:
                    mock_mem.return_value = MagicMock(total=32 * 1024**3)
                    specs = scan_machine()
                    assert specs.vram_mb is None
                    assert specs.gpu_name == ""

    def test_scan_machine_darwin(self):
        with patch("godspeed.evolution.hardware._get_cached_vram", return_value=None):
            with patch("godspeed.evolution.hardware._detect_gpu_name", return_value=""):
                with patch("psutil.virtual_memory") as mock_mem:
                    with patch("platform.system", return_value="Darwin"):
                        mock_mem.return_value = MagicMock(total=16 * 1024**3)
                        specs = scan_machine()
                        assert specs.platform == "darwin"


class TestDetectNvidiaSmiEdgeCases:
    def test_empty_output(self):
        with patch("shutil.which", return_value="nvidia-smi"):
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = "\n"
            with patch("subprocess.run", return_value=mock_result):
                vram = _detect_nvidia_smi()
                assert vram is None

    def test_timeout_error(self):
        import subprocess

        with patch("shutil.which", return_value="nvidia-smi"):
            with patch(
                "subprocess.run",
                side_effect=subprocess.TimeoutExpired("nvidia-smi", 10),
            ):
                vram = _detect_nvidia_smi()
                assert vram is None

    def test_os_error(self):
        with patch("shutil.which", return_value="nvidia-smi"):
            with patch("subprocess.run", side_effect=OSError("broken")):
                vram = _detect_nvidia_smi()
                assert vram is None

    def test_non_numeric_output(self):
        with patch("shutil.which", return_value="nvidia-smi"):
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = "not_a_number\n"
            with patch("subprocess.run", return_value=mock_result):
                vram = _detect_nvidia_smi()
                assert vram is None

    def test_decimal_value(self):
        with patch("shutil.which", return_value="nvidia-smi"):
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = "8192.5\n"
            with patch("subprocess.run", return_value=mock_result):
                vram = _detect_nvidia_smi()
                assert vram == 8192

    def test_missing_nvidia_smi_binary(self):
        with patch("shutil.which", return_value=None):
            vram = _detect_nvidia_smi()
            assert vram is None


class TestDetectJetsonEdgeCases:
    def test_valid_meminfo(self):
        meminfo = "MemTotal: 8388608 kB\nMemFree: 4194304 kB\nMemAvailable: 6291456 kB\n"
        mock_file = MagicMock()
        mock_file.__iter__.return_value = iter(meminfo.splitlines(True))
        mock_file.__enter__.return_value = mock_file
        mock_file.__exit__.return_value = None
        with patch("builtins.open", return_value=mock_file):
            vram = _detect_jetson()
            assert vram is not None
            assert vram > 0

    def test_memavailable_missing(self):
        meminfo = "MemTotal: 8388608 kB\nMemFree: 4194304 kB\n"
        mock_file = MagicMock()
        mock_file.__iter__.return_value = iter(meminfo.splitlines(True))
        mock_file.__enter__.return_value = mock_file
        mock_file.__exit__.return_value = None
        with patch("builtins.open", return_value=mock_file):
            vram = _detect_jetson()
            assert vram is None

    def test_corrupted_meminfo(self):
        meminfo = "MemAvailable: broken kB\n"
        mock_file = MagicMock()
        mock_file.__iter__.return_value = iter(meminfo.splitlines(True))
        mock_file.__enter__.return_value = mock_file
        mock_file.__exit__.return_value = None
        with patch("builtins.open", return_value=mock_file):
            vram = _detect_jetson()
            assert vram is None

    def test_permission_error(self):
        with patch("builtins.open", side_effect=PermissionError("no access")):
            vram = _detect_jetson()
            assert vram is None


class TestDetectGpuNameEdgeCases:
    def test_nvidia_smi_failure_returns_empty(self):
        with patch("shutil.which", return_value="nvidia-smi"):
            mock_result = MagicMock()
            mock_result.returncode = 1
            mock_result.stdout = ""
            with patch("subprocess.run", return_value=mock_result):
                name = _detect_gpu_name()
                assert name == ""

    def test_nvidia_smi_timeout(self):
        import subprocess

        with patch("shutil.which", return_value="nvidia-smi"):
            with patch(
                "subprocess.run",
                side_effect=subprocess.TimeoutExpired("nvidia-smi", 5),
            ):
                name = _detect_gpu_name()
                assert name == ""

    def test_nvidia_smi_os_error(self):
        with patch("shutil.which", return_value="nvidia-smi"):
            with patch("subprocess.run", side_effect=OSError("broken")):
                name = _detect_gpu_name()
                assert name == ""


class TestGetCachedVramThreadSafety:
    def test_double_check_locking(self):
        import godspeed.evolution.hardware as hw_module

        old_vram = hw_module._cached_vram
        old_checked = hw_module._cached_vram_checked
        try:
            hw_module._cached_vram = None
            hw_module._cached_vram_checked = False
            with patch(
                "godspeed.evolution.hardware.detect_vram_mb", return_value=8000
            ) as mock_detect:
                vram = _get_cached_vram()
                assert vram == 8000
                assert mock_detect.call_count == 1
                assert hw_module._cached_vram_checked is True
        finally:
            hw_module._cached_vram = old_vram
            hw_module._cached_vram_checked = old_checked

    def test_cache_returns_without_redetection(self):
        import godspeed.evolution.hardware as hw_module

        old_vram = hw_module._cached_vram
        old_checked = hw_module._cached_vram_checked
        try:
            hw_module._cached_vram = 12000
            hw_module._cached_vram_checked = True
            with patch(
                "godspeed.evolution.hardware.detect_vram_mb"
            ) as mock_detect:
                vram = _get_cached_vram()
                assert vram == 12000
                mock_detect.assert_not_called()
        finally:
            hw_module._cached_vram = old_vram
            hw_module._cached_vram_checked = old_checked


class TestRecommendedNumCandidatesFallsThrough:
    def test_zero_vram_falls_through_to_one(self):
        with patch("godspeed.evolution.hardware._get_cached_vram", return_value=0):
            assert recommended_num_candidates() == 1

    def test_ollama_model_uses_vram_path(self):
        with patch("godspeed.evolution.hardware._get_cached_vram", return_value=8000):
            n = recommended_num_candidates("ollama/qwen2.5-coder:14b")
            assert n >= 1


class TestRecommendedMaxEvalCasesFallsThrough:
    def test_zero_vram_falls_through_to_one(self):
        with patch("godspeed.evolution.hardware._get_cached_vram", return_value=0):
            assert recommended_max_eval_cases() == 1

    def test_ollama_model_uses_vram_path_eval(self):
        with patch("godspeed.evolution.hardware._get_cached_vram", return_value=8000):
            n = recommended_max_eval_cases("ollama/qwen2.5-coder:14b")
            assert n >= 1

    def test_api_model_returns_five_eval_cases(self):
        n = recommended_max_eval_cases("anthropic/claude-sonnet-4-20250514")
        assert n == 5

    def test_none_vram_returns_two_eval_cases(self):
        with patch("godspeed.evolution.hardware._get_cached_vram", return_value=None):
            assert recommended_max_eval_cases() == 2
