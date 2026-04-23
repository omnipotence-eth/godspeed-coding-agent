"""Tests for the optional speech I/O subsystem.

The real audio-device + Whisper paths are NOT exercised here — they
require hardware + ~150MB model download. These tests cover the bits
that CAN be validated offline: availability gating, command routing,
and clean error messages when the extra isn't installed.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from godspeed.agent.conversation import Conversation
from godspeed.speech import availability
from godspeed.tui.commands import Commands


class TestAvailability:
    def test_is_available_when_both_halves_present(self) -> None:
        # Simulate every required module being importable.
        with patch.object(availability, "_importable", return_value=True):
            assert availability.is_available() is True
            assert availability.stt_available() is True
            assert availability.tts_available() is True

    def test_is_unavailable_when_stt_missing(self) -> None:
        def _fake(name: str) -> bool:
            return name not in {"faster_whisper", "sounddevice"}

        with patch.object(availability, "_importable", side_effect=_fake):
            assert availability.stt_available() is False
            assert availability.is_available() is False

    def test_is_unavailable_when_no_tts_backend(self) -> None:
        def _fake(name: str) -> bool:
            return name not in {"piper", "pyttsx3"}

        with patch.object(availability, "_importable", side_effect=_fake):
            assert availability.tts_available() is False
            assert availability.is_available() is False

    def test_missing_extras_message_names_missing_pieces(self) -> None:
        with patch.object(availability, "_importable", return_value=False):
            msg = availability.missing_extras_message()
        assert "pip install" in msg
        assert "godspeed-coding-agent[speech]" in msg
        # Every STT-required pkg appears by name in the missing list.
        for pkg in availability._REQUIRED_FOR_STT:
            assert pkg in msg
        # TTS fallback hint mentions both options.
        assert "piper" in msg.lower()
        assert "pyttsx3" in msg.lower()

    def test_ready_message_when_all_present(self) -> None:
        with patch.object(availability, "_importable", return_value=True):
            assert availability.missing_extras_message() == "Speech I/O is ready."


@pytest.fixture
def commands(tmp_path: Path) -> Commands:
    conv = Conversation("sys", max_tokens=100_000)
    llm = MagicMock()
    llm.model = "m"
    llm.fallback_models = []
    llm.total_input_tokens = 0
    llm.total_output_tokens = 0
    return Commands(
        conversation=conv,
        llm_client=llm,
        permission_engine=MagicMock(),
        audit_trail=None,
        session_id="s",
        cwd=tmp_path,
    )


class TestListenCommand:
    def test_missing_extras_shows_install_hint(self, commands: Commands) -> None:
        with patch("godspeed.speech.availability.is_available", return_value=False):
            result = commands.dispatch("/listen")
        assert result is not None and result.handled

    def test_rejects_invalid_duration(self, commands: Commands) -> None:
        with patch("godspeed.speech.availability.is_available", return_value=True):
            result = commands.dispatch("/listen abc")
        assert result is not None and result.handled

    def test_rejects_out_of_range_duration(self, commands: Commands) -> None:
        with patch("godspeed.speech.availability.is_available", return_value=True):
            result = commands.dispatch("/listen 200")  # > 120 cap
        assert result is not None and result.handled

    def test_transcription_injects_as_user_message(self, commands: Commands) -> None:
        with (
            patch("godspeed.speech.availability.is_available", return_value=True),
            patch("godspeed.speech.stt.record_and_transcribe", return_value="hello agent"),
        ):
            result = commands.dispatch("/listen 5")
        assert result is not None
        assert result.handled is False  # lets agent_loop run with the injected message
        user_messages = [m for m in commands._conversation.messages if m.get("role") == "user"]
        assert any("hello agent" in m.get("content", "") for m in user_messages)

    def test_empty_transcription_warns(self, commands: Commands) -> None:
        with (
            patch("godspeed.speech.availability.is_available", return_value=True),
            patch("godspeed.speech.stt.record_and_transcribe", return_value=""),
        ):
            result = commands.dispatch("/listen")
        assert result is not None and result.handled
        # No user message injected.
        user_messages = [m for m in commands._conversation.messages if m.get("role") == "user"]
        assert not user_messages

    def test_stt_raises_does_not_crash_tui(self, commands: Commands) -> None:
        with (
            patch("godspeed.speech.availability.is_available", return_value=True),
            patch(
                "godspeed.speech.stt.record_and_transcribe",
                side_effect=RuntimeError("audio device busy"),
            ),
        ):
            result = commands.dispatch("/listen")
        assert result is not None and result.handled


class TestSpeakCommand:
    def test_toggle_on_enables_flag(self, commands: Commands) -> None:
        with patch("godspeed.speech.availability.is_available", return_value=True):
            commands.dispatch("/speak on")
        assert commands.speak_enabled is True

    def test_toggle_off_disables_flag(self, commands: Commands) -> None:
        commands.speak_enabled = True
        commands.dispatch("/speak off")
        assert commands.speak_enabled is False

    def test_toggle_on_without_extra_shows_hint(self, commands: Commands) -> None:
        with patch("godspeed.speech.availability.is_available", return_value=False):
            commands.dispatch("/speak on")
        # Flag should NOT flip when the extra is missing — can't
        # enable something that doesn't work.
        assert commands.speak_enabled is False

    def test_bare_speak_shows_current_state(self, commands: Commands) -> None:
        result = commands.dispatch("/speak")
        assert result is not None and result.handled

    def test_literal_text_speaks_without_toggling(self, commands: Commands) -> None:
        commands.speak_enabled = False
        with (
            patch("godspeed.speech.availability.is_available", return_value=True),
            patch("godspeed.speech.tts.speak") as mock_speak,
        ):
            result = commands.dispatch('/speak "hello world"')
        assert result is not None and result.handled
        mock_speak.assert_called_once()
        spoken = mock_speak.call_args.args[0]
        assert "hello world" in spoken
        # Toggle state unchanged by the literal-text form.
        assert commands.speak_enabled is False

    def test_literal_text_raise_does_not_crash(self, commands: Commands) -> None:
        with (
            patch("godspeed.speech.availability.is_available", return_value=True),
            patch("godspeed.speech.tts.speak", side_effect=RuntimeError("no output")),
        ):
            result = commands.dispatch('/speak "test"')
        assert result is not None and result.handled
