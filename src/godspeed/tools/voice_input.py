"""Voice input tool for speech-to-text."""

from __future__ import annotations

import logging
from typing import Any

from godspeed.tools.base import RiskLevel, Tool, ToolContext, ToolResult

logger = logging.getLogger(__name__)


class VoiceInputTool(Tool):
    """Convert speech to text using the microphone.

    Uses the system microphone to capture audio and convert
    to text. Useful for hands-free coding.
    """

    produces_diff = False

    @property
    def name(self) -> str:
        return "voice_input"

    @property
    def description(self) -> str:
        return (
            "Record audio from microphone and convert to text. "
            "Press start to begin recording, stop to finish. "
            "Useful for hands-free coding or dictating code."
        )

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.LOW

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["start", "stop", "status"],
                    "description": "Action to perform",
                },
                "duration": {
                    "type": "integer",
                    "description": "Max recording duration in seconds",
                },
            },
            "required": ["action"],
        }

    async def execute(
        self,
        tool_context: ToolContext,
        action: str = "status",
        duration: int = 10,
    ) -> ToolResult:
        """Execute voice input action."""
        import os
        import tempfile

        if action == "status":
            return ToolResult.ok("Voice input ready. Use action=start to begin recording.")

        elif action == "start":
            try:
                import numpy as np
                import sounddevice as sd

                # Record audio
                audio_data = []
                recording = True

                def callback(indata, frames, time, status):
                    if status:
                        logger.warning("Audio status: %s", status)
                    audio_data.append(indata.copy())

                with sd.InputStream(channels=1, samplerate=16000, callback=callback):
                    await asyncio.sleep(duration)

                if not audio_data:
                    return ToolResult.failure("No audio recorded")

                # Save to temp file
                audio_np = np.concatenate(audio_data)
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    temp_file = f.name

                try:
                    import wave

                    with wave.open(temp_file, "w") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(16000)
                        wf.writeframes(audio_np.tobytes())

                    # Try speech recognition
                    try:
                        import speech_recognition as sr

                        r = sr.Recognizer()
                        with sr.AudioFile(temp_file) as source:
                            audio = r.record(source)
                        text = r.recognize_google(audio)
                        return ToolResult.ok(f"Transcribed: {text}")

                    except ImportError:
                        # Fallback: return audio file path
                        return ToolResult.ok(f"Audio saved to: {temp_file}")

                finally:
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)

            except ImportError:
                return ToolResult.failure(
                    "Voice input requires 'sounddevice' and 'SpeechRecognition'. "
                    "Install with: pip install sounddevice SpeechRecognition"
                )
            except Exception as exc:
                return ToolResult.failure(f"Recording failed: {exc}")

        return ToolResult.failure(f"Unknown action: {action}")


import asyncio
