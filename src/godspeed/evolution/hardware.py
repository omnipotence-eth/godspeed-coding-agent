"""Hardware-aware model selection for evolution — auto-detect VRAM, pick largest fitting model.

Handles everything from RTX 5070 Ti (16GB dedicated) to Jetson Orin Nano (8GB shared RAM).
Falls back gracefully when VRAM detection fails or Ollama is unavailable.
"""

from __future__ import annotations

import dataclasses
import logging
import os as _os
import platform
import shutil
import subprocess

logger = logging.getLogger(__name__)

# Cached VRAM value — subprocess call is expensive, so detect once per process.
# A dict key is used instead of a simple module-level variable so the sentinel
# "not yet checked" (None) is distinguishable from "checked but no GPU found"
# (also returned as None by detect_vram_mb).
_cached_vram: int | None = None
_cached_vram_checked: bool = False


def _get_cached_vram() -> int | None:
    """Return cached VRAM, running detection on first call."""
    global _cached_vram, _cached_vram_checked
    if not _cached_vram_checked:
        _cached_vram = detect_vram_mb()
        _cached_vram_checked = True
    return _cached_vram


# ---------------------------------------------------------------------------
# VRAM tiers → model selection
# ---------------------------------------------------------------------------

# Ordered largest-first. Each entry: (min_vram_mb, model, description)
# VRAM thresholds account for ~1GB overhead from Ollama + OS.
# Models have been benchmarked for coding agent performance (SWE-bench, tool-calling).
MODEL_TIERS: list[tuple[int, str, str]] = [
    (12_000, "ollama/devstral-small-2:24b", "24B — 65.8% SWE-bench, 384K ctx (15GB)"),
    (8_000, "ollama/qwen2.5-coder:14b", "14B — strong code repair, 32K ctx (9GB)"),
    (5_000, "ollama/rnj-1:8b", "8B — #1 tool-calling, 32K ctx (5.1GB)"),
    (3_000, "ollama/cogito:14b", "14B — hybrid reasoning, 128K ctx"),
    (1_500, "ollama/qwen2.5:1.5b", "1.5B — absolute fallback (1.5GB)"),
]

# Absolute fallback if nothing fits or detection fails
FALLBACK_MODEL = "ollama/rnj-1:8b"

# num_candidates scaling by available VRAM
CANDIDATES_BY_VRAM: list[tuple[int, int]] = [
    (10_000, 5),  # 10GB+ → up to 5 candidates
    (6_000, 3),  # 6GB+ → up to 3 candidates
    (3_000, 2),  # 3GB+ → 2 candidates
    (0, 1),  # anything less → 1 candidate
]

# Max concurrent A/B test cases for fitness evaluation
MAX_EVAL_CASES_BY_VRAM: list[tuple[int, int]] = [
    (10_000, 5),
    (6_000, 3),
    (3_000, 2),
    (0, 1),
]


# ---------------------------------------------------------------------------
# VRAM detection
# ---------------------------------------------------------------------------


def detect_vram_mb() -> int | None:
    """Detect available GPU VRAM in MB. Returns None if detection fails.

    Tries nvidia-smi first (discrete NVIDIA GPUs), then tegrastats-style
    detection (Jetson), then returns None.
    """
    vram = _detect_nvidia_smi()
    if vram is not None:
        return vram

    vram = _detect_jetson()
    if vram is not None:
        return vram

    return None


def _detect_nvidia_smi() -> int | None:
    """Query nvidia-smi for free VRAM in MB."""
    if not shutil.which("nvidia-smi"):
        return None

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return None

        # Take the first GPU's free memory
        for line in result.stdout.strip().splitlines():
            line = line.strip()
            if line:
                return int(float(line))
    except (subprocess.TimeoutExpired, ValueError, OSError) as exc:
        logger.debug("nvidia-smi detection failed: %s", exc)

    return None


def _detect_jetson() -> int | None:
    """Detect available memory on Jetson (shared RAM architecture).

    On Jetson, GPU and CPU share the same RAM pool. We read /proc/meminfo
    for available memory and apply a conservative factor since GPU workloads
    compete with the OS and CPU processes.
    """
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    # Value is in kB
                    kb = int(line.split()[1])
                    # Jetson shares RAM — use 60% of available as VRAM budget
                    return int((kb / 1024) * 0.6)
    except (OSError, ValueError, IndexError):
        pass

    return None


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------


def select_evolution_model(configured_model: str = "") -> str:
    """Select the best evolution model based on config and available hardware.

    Priority:
    1. Explicit user configuration (non-empty configured_model)
    2. Auto-detect VRAM and pick largest fitting model
    3. Fallback to smallest model (qwen2.5:1.5b)

    Args:
        configured_model: User-configured model string (from evolution_model config).

    Returns:
        LiteLLM-compatible model string.
    """
    if configured_model:
        logger.debug("Using configured evolution model=%s", configured_model)
        return configured_model

    vram = _get_cached_vram()
    if vram is None:
        logger.info("Could not detect VRAM — defaulting to %s", FALLBACK_MODEL)
        return FALLBACK_MODEL

    logger.info("Detected %d MB available VRAM for evolution", vram)

    for min_vram, model, desc in MODEL_TIERS:
        if vram >= min_vram:
            logger.info("Selected evolution model=%s (%s)", model, desc)
            return model

    logger.info("Low VRAM (%d MB) — using fallback %s", vram, FALLBACK_MODEL)
    return FALLBACK_MODEL


def recommended_num_candidates(configured_model: str = "") -> int:
    """Return recommended number of mutation candidates based on available VRAM.

    Fewer candidates = less VRAM pressure and faster evolution cycles.
    """
    if configured_model and not configured_model.startswith("ollama/"):
        # API models have no local VRAM constraint
        return 5

    vram = _get_cached_vram()
    if vram is None:
        return 2  # Conservative default

    for min_vram, candidates in CANDIDATES_BY_VRAM:
        if vram >= min_vram:
            return candidates

    return 1


def recommended_max_eval_cases(configured_model: str = "") -> int:
    """Return recommended max A/B test cases for fitness evaluation."""
    if configured_model and not configured_model.startswith("ollama/"):
        return 5

    vram = _get_cached_vram()
    if vram is None:
        return 2

    for min_vram, cases in MAX_EVAL_CASES_BY_VRAM:
        if vram >= min_vram:
            return cases

    return 1


def is_low_memory() -> bool:
    """Check if running in a low-memory environment (< 6GB VRAM).

    Low-memory mode: fewer candidates, smaller batch sizes, streaming traces.
    """
    vram = _get_cached_vram()
    if vram is None:
        return True  # Assume constrained if we can't detect
    return vram < 6_000


# ---------------------------------------------------------------------------
# Machine scan — recommend the optimal model for each tier
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class MachineSpecs:
    """Machine hardware specifications for model selection."""

    platform: str          # "windows", "linux", "darwin"
    vram_mb: int | None    # detected GPU VRAM or None
    ram_gb: float          # system RAM in GB
    cpu_cores: int         # logical CPU cores
    gpu_name: str          # GPU name or empty string


def scan_machine() -> MachineSpecs:
    """Scan the current machine for hardware specifications.

    Returns a MachineSpecs that can be used to recommend optimal models.
    """
    vram = _get_cached_vram()
    gpu_name = _detect_gpu_name()

    # System RAM
    ram_gb = 0.0
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024**3)
    except ImportError:
        pass

    cpu_cores = _os.cpu_count() or 1

    return MachineSpecs(
        platform=platform.system().lower(),
        vram_mb=vram,
        ram_gb=round(ram_gb, 1),
        cpu_cores=cpu_cores,
        gpu_name=gpu_name,
    )


def _detect_gpu_name() -> str:
    """Detect the GPU name string."""
    if shutil.which("nvidia-smi"):
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip().splitlines()[0].strip()
        except (subprocess.TimeoutExpired, OSError):
            pass
    return ""


def recommend_models_for_machine(specs: MachineSpecs | None = None) -> dict[str, str | None]:
    """Recommend the best model for each preset tier based on machine specs.

    Returns a dict mapping preset tier names to recommended model strings.
    Models that don't fit the machine are marked as None.

    Example:
        {
            "fast": "ollama/rnj-1:8b",
            "balanced": "ollama/qwen2.5-coder:14b",
            "quality": None,  # doesn't fit
            "cloud": "nvidia_nim/qwen/qwen3.5-397b-a17b",
            "frontier": "claude-sonnet-4-20250514",
        }
    """
    if specs is None:
        specs = scan_machine()

    vram = specs.vram_mb
    ram = specs.ram_gb

    # 1GB overhead for OS + Ollama runtime
    overhead = 1024

    recommendations: dict[str, str | None] = {}

    # Local tiers — depend on VRAM (GPU) or RAM (CPU inference)
    # GPU mode: use VRAM
    # CPU mode: use 60% of system RAM as budget
    if vram is not None and vram > 0:
        budget = vram - overhead
        mode = "GPU"
    else:
        budget = int(ram * 1024 * 0.6)  # 60% of RAM for CPU inference
        mode = "CPU"

    logger.info("Model selection mode=%s budget=%dMB vram=%s ram=%.1fGB", mode, budget, vram, ram)

    # Define local model tiers: (min_mb, model_name, description)
    local_tiers: list[tuple[int, str, str]] = [
        (12_000, "quality", "ollama/devstral-small-2:24b"),
        (8_000, "balanced", "ollama/qwen2.5-coder:14b"),
        (6_000, "balanced_alt", "ollama/deepseek-coder-v2:16b"),
        (4_000, "fast", "ollama/rnj-1:8b"),
        (3_000, "fast_alt", "ollama/qwen2.5-coder:7b"),
        (1_500, "fallback", "ollama/qwen2.5:1.5b"),
    ]

    selected_fast = "ollama/qwen2.5:1.5b"
    selected_balanced = None
    selected_quality = None

    for min_mb, tier_name, model in local_tiers:
        if budget >= min_mb:
            if tier_name == "quality" and selected_quality is None:
                selected_quality = model
            elif tier_name in ("balanced", "balanced_alt") and selected_balanced is None:
                selected_balanced = model
            elif tier_name == "fast" and selected_fast == model:
                pass  # already set
            elif tier_name in ("fast_alt", "fallback"):
                selected_fast = model

    # Walk back if no balanced: balanced = fast model
    if selected_balanced is None:
        selected_balanced = selected_fast
        selected_quality = None

    recommendations["fast"] = selected_fast
    recommendations["balanced"] = selected_balanced
    recommendations["quality"] = selected_quality

    # Cloud and frontier tiers always available (API-based, no local GPU needed)
    recommendations["cloud"] = "nvidia_nim/qwen/qwen3.5-397b-a17b"
    recommendations["frontier"] = "claude-sonnet-4-20250514"

    return recommendations


def format_machine_report(specs: MachineSpecs | None = None) -> str:
    """Generate a human-readable machine scan report with recommendations.

    Returns a formatted string suitable for display in the terminal.
    """
    if specs is None:
        specs = scan_machine()

    recs = recommend_models_for_machine(specs)

    lines = [
        "=" * 62,
        "  GODSPEED MACHINE SCAN",
        "=" * 62,
        f"  Platform:     {specs.platform}",
        f"  CPU Cores:    {specs.cpu_cores}",
        f"  System RAM:   {specs.ram_gb:.1f} GB",
    ]

    if specs.gpu_name:
        lines.append(f"  GPU:          {specs.gpu_name}")
    if specs.vram_mb:
        lines.append(f"  VRAM:         {specs.vram_mb / 1024:.1f} GB free")
    else:
        lines.append("  VRAM:         not detected (CPU-only mode)")

    lines.append("")
    lines.append("  RECOMMENDED MODELS:")
    lines.append("  " + "-" * 42)

    tier_descriptions = {
        "fast": "Fast (local, low VRAM)",
        "balanced": "Balanced (local, medium VRAM)",
        "quality": "Quality (local, high VRAM)",
        "cloud": "Cloud (NVIDIA NIM free tier)",
        "frontier": "Frontier (Claude, best quality)",
    }

    for tier in ("fast", "balanced", "quality", "cloud", "frontier"):
        model = recs.get(tier)
        desc = tier_descriptions.get(tier, tier)
        if model:
            status = "✓"
            lines.append(f"  {status} {desc:35s} ->  {model}")
        else:
            status = "✗"
            lines.append(f"  {status} {desc:35s} ->  (insufficient VRAM)")

    lines.append("")
    lines.append("  Use --preset <tier> or /model <tier> to switch.")

    return "\n".join(lines)
