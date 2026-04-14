"""Hardware-aware model selection for evolution — auto-detect VRAM, pick largest fitting model.

Handles everything from RTX 5070 Ti (16GB dedicated) to Jetson Orin Nano (8GB shared RAM).
Falls back gracefully when VRAM detection fails or Ollama is unavailable.
"""

from __future__ import annotations

import logging
import shutil
import subprocess

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# VRAM tiers → model selection
# ---------------------------------------------------------------------------

# Ordered largest-first. Each entry: (min_vram_mb, model, description)
# VRAM thresholds account for ~1GB overhead from Ollama + OS.
MODEL_TIERS: list[tuple[int, str, str]] = [
    (10_000, "ollama/gemma3:12b", "12B — needs ~10GB free VRAM"),
    (6_000, "ollama/gemma3:4b", "4B — needs ~6GB free VRAM"),
    (3_000, "ollama/qwen2.5:3b", "3B — needs ~3GB free VRAM"),
    (1_500, "ollama/qwen2.5:1.5b", "1.5B — needs ~1.5GB free VRAM"),
]

# Absolute fallback if nothing fits or detection fails
FALLBACK_MODEL = "ollama/qwen2.5:1.5b"

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

    vram = detect_vram_mb()
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

    vram = detect_vram_mb()
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

    vram = detect_vram_mb()
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
    vram = detect_vram_mb()
    if vram is None:
        return True  # Assume constrained if we can't detect
    return vram < 6_000
