"""Speech-to-text — record from the default mic and transcribe with Whisper."""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from godspeed.speech.availability import stt_available

if TYPE_CHECKING:
    from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

# faster-whisper model size. "tiny" fits in ~75MB VRAM, ~200ms/s on
# CPU, ~20ms/s on a modern consumer GPU. "base" is ~150MB and a bit
# more accurate; "small" is 244MB. Tiny is the default because first-
# run download time matters more than last-10%-accuracy for the
# push-to-talk use case.
DEFAULT_MODEL_SIZE = "tiny.en"

# Sample rate Whisper expects. 16kHz mono f32.
_WHISPER_SAMPLE_RATE = 16_000


class _SttUnavailableError(RuntimeError):
    """Raised when an STT function is called without deps installed."""


def _lazy_whisper_model(model_size: str, device: str) -> WhisperModel:
    """Import faster-whisper + instantiate the model. Cached per-process."""
    from faster_whisper import WhisperModel

    return WhisperModel(model_size, device=device, compute_type="default")


def record_from_mic(duration_seconds: float, output_wav: Path | None = None) -> bytes | Path:
    """Record ``duration_seconds`` of 16kHz mono audio from the default input.

    Returns the raw int16 PCM bytes (when ``output_wav`` is None) or
    the path to a written .wav file (when ``output_wav`` is provided).

    Small intentional blocking API — push-to-talk is a user-in-the-loop
    pattern where "block for exactly N seconds" is the expected
    contract. Wrap in ``asyncio.to_thread`` at the call site if the
    caller needs async.

    Raises:
        _SttUnavailableError: if ``sounddevice`` isn't installed.
    """
    if not stt_available():
        raise _SttUnavailableError(
            "sounddevice / numpy / faster-whisper not installed — "
            "install with: pip install 'godspeed-coding-agent[speech]'"
        )

    import numpy as np
    import sounddevice as sd

    logger.info("Recording %.1fs from default mic @ %d Hz", duration_seconds, _WHISPER_SAMPLE_RATE)
    # int16 because that's the standard PCM wire format; f32 works too
    # but int16 halves the memory and Whisper converts internally.
    n_frames = int(duration_seconds * _WHISPER_SAMPLE_RATE)
    audio = sd.rec(n_frames, samplerate=_WHISPER_SAMPLE_RATE, channels=1, dtype="int16")
    sd.wait()  # block until recording completes

    if output_wav is None:
        return bytes(audio)

    import wave

    with wave.open(str(output_wav), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16 = 2 bytes
        wf.setframerate(_WHISPER_SAMPLE_RATE)
        wf.writeframes(np.asarray(audio, dtype=np.int16).tobytes())
    return output_wav


def transcribe_wav(
    wav_path: Path,
    model_size: str = DEFAULT_MODEL_SIZE,
    device: str = "auto",
) -> str:
    """Transcribe a .wav file to text.

    ``device`` follows faster-whisper's convention: ``"auto"`` picks
    CUDA if available (RTX 5070 Ti will use it), falling back to CPU.

    Raises:
        _SttUnavailableError: if ``faster-whisper`` isn't installed.
    """
    if not stt_available():
        raise _SttUnavailableError(
            "faster-whisper not installed — "
            "install with: pip install 'godspeed-coding-agent[speech]'"
        )

    model = _lazy_whisper_model(model_size, device)
    segments, info = model.transcribe(str(wav_path), beam_size=1)
    text_parts: list[str] = []
    for seg in segments:
        text_parts.append(seg.text)
    full_text = "".join(text_parts).strip()
    logger.info(
        "Transcribed duration=%.1fs language=%s chars=%d",
        info.duration,
        info.language,
        len(full_text),
    )
    return full_text


def record_and_transcribe(
    duration_seconds: float = 8.0,
    model_size: str = DEFAULT_MODEL_SIZE,
    device: str = "auto",
) -> str:
    """End-to-end: record from mic, transcribe, return text.

    Writes a temporary .wav that's deleted after transcription.

    Raises:
        _SttUnavailableError: if the speech extra isn't installed.
    """
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        record_from_mic(duration_seconds, output_wav=tmp_path)
        return transcribe_wav(tmp_path, model_size=model_size, device=device)
    finally:
        with contextlib.suppress(OSError):
            tmp_path.unlink()
