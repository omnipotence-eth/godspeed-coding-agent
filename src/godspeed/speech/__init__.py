"""Godspeed speech I/O — push-to-talk STT + streaming TTS.

Opt-in via ``pip install godspeed[speech]``. Without the extra
installed, ``is_available()`` returns False and the ``/listen`` /
``/speak`` slash commands tell the user how to enable it instead of
failing with ``ModuleNotFoundError``.

Architecture:

- :mod:`godspeed.speech.stt` — Whisper-based speech-to-text via
  ``faster-whisper`` on the default audio input device.
- :mod:`godspeed.speech.tts` — Piper-based text-to-speech on the
  default audio output device. Graceful fallback to ``pyttsx3`` if
  Piper isn't present (same API surface either way).

The slash-command adapter (``godspeed.tui.commands._cmd_listen`` and
``_cmd_speak``) is kept thin; business logic lives here so it can be
unit-tested without the TUI.
"""

from __future__ import annotations

from godspeed.speech.availability import is_available, missing_extras_message

__all__ = ["is_available", "missing_extras_message"]
