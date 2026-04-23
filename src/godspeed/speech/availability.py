"""Runtime dependency check for the optional ``[speech]`` extra."""

from __future__ import annotations

import importlib.util

# Required for any speech I/O. If ALL of these are importable,
# ``is_available()`` returns True. Missing any one reveals the
# install-instructions error path in the slash commands.
#
# ``pyttsx3`` is accepted as a TTS fallback so Windows users who
# can't easily get Piper running still get working ``/speak``.
_REQUIRED_FOR_STT: tuple[str, ...] = ("faster_whisper", "sounddevice", "numpy")
_REQUIRED_FOR_TTS_ANY: tuple[str, ...] = ("piper", "pyttsx3")


def _importable(name: str) -> bool:
    """Return True if the given module can be imported (spec lookup only)."""
    return importlib.util.find_spec(name) is not None


def stt_available() -> bool:
    """True when every STT-path dependency is installed."""
    return all(_importable(pkg) for pkg in _REQUIRED_FOR_STT)


def tts_available() -> bool:
    """True when at least one TTS backend is installed."""
    return any(_importable(pkg) for pkg in _REQUIRED_FOR_TTS_ANY)


def is_available() -> bool:
    """True when BOTH halves of speech I/O are usable."""
    return stt_available() and tts_available()


def missing_extras_message() -> str:
    """Human-readable hint listing which pieces are missing, for logs/UI."""
    missing: list[str] = []
    if not stt_available():
        missing.extend(f"  - {pkg} (STT)" for pkg in _REQUIRED_FOR_STT if not _importable(pkg))
    if not tts_available():
        missing.append(
            "  - one of {piper-tts, pyttsx3} (TTS). Piper gives the best "
            "quality; pyttsx3 is the cross-platform fallback."
        )
    if not missing:
        return "Speech I/O is ready."
    lines = [
        "Speech I/O is not available. Install the optional extra:",
        "  pip install 'godspeed-coding-agent[speech]'",
        "",
        "Missing:",
        *missing,
    ]
    return "\n".join(lines)
