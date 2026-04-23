"""Text-to-speech — play agent responses through the default output device.

Uses Piper (high quality, local, ONNX-based) when available, with
``pyttsx3`` as a cross-platform fallback. API shape is identical so
the slash-command adapter doesn't need to care which backend is live.
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path

from godspeed.speech.availability import tts_available

logger = logging.getLogger(__name__)


class _TtsUnavailableError(RuntimeError):
    """Raised when TTS is called without any backend installed."""


def _pick_backend() -> str:
    """Return the name of the TTS backend that will actually run.

    Piper first (better quality, local), pyttsx3 second (ubiquitous).
    """
    if importlib.util.find_spec("piper") is not None:
        return "piper"
    if importlib.util.find_spec("pyttsx3") is not None:
        return "pyttsx3"
    return ""


def _speak_piper(text: str, voice_model: Path | None) -> None:
    """Synthesize via Piper + play via sounddevice."""
    import numpy as np
    import sounddevice as sd
    from piper import PiperVoice  # type: ignore[import-not-found]

    if voice_model is None:
        raise _TtsUnavailableError(
            "Piper requires a voice model path — set GODSPEED_PIPER_VOICE "
            "or pass voice_model=Path('...'). See "
            "https://github.com/rhasspy/piper#voices"
        )
    voice = PiperVoice.load(str(voice_model))
    # Piper yields chunks of int16 samples at voice.config.sample_rate.
    # The streaming API has moved between piper-tts versions — the
    # older ``synthesize_stream_raw`` and the newer ``synthesize`` both
    # yield raw int16 bytes. Probe at runtime.
    synth = getattr(voice, "synthesize_stream_raw", None) or voice.synthesize  # type: ignore[attr-defined]
    audio_chunks: list[bytes] = []
    for chunk in synth(text):
        audio_chunks.append(chunk)
    if not audio_chunks:
        return
    raw = b"".join(audio_chunks)
    samples = np.frombuffer(raw, dtype=np.int16)
    sd.play(samples, samplerate=voice.config.sample_rate)
    sd.wait()


def _speak_pyttsx3(text: str) -> None:
    """Synthesize + play via pyttsx3 (OS-native backend)."""
    import pyttsx3  # type: ignore[import-not-found]

    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()


def speak(text: str, voice_model: Path | None = None) -> None:
    """Speak ``text`` through the default output device. Blocks until done.

    Picks the best available backend. When both are installed, Piper
    wins unless ``voice_model`` is missing AND no default is
    available, in which case we fall through to pyttsx3.

    Raises:
        _TtsUnavailableError: if no backend is installed or Piper is
            selected but no voice model is available.
    """
    if not text or not text.strip():
        return
    if not tts_available():
        raise _TtsUnavailableError(
            "No TTS backend installed. Install with: pip install 'godspeed-coding-agent[speech]'"
        )

    backend = _pick_backend()
    if backend == "piper":
        try:
            _speak_piper(text, voice_model)
            return
        except _TtsUnavailableError as exc:
            # No voice model — try pyttsx3 if it's around.
            if importlib.util.find_spec("pyttsx3") is not None:
                logger.info("Piper voice not available, falling back to pyttsx3: %s", exc)
            else:
                raise
    if importlib.util.find_spec("pyttsx3") is not None:
        _speak_pyttsx3(text)
        return

    # Unreachable given the tts_available() check above, but defensive.
    raise _TtsUnavailableError("TTS backend selection failed — no backend usable.")
