"""Tests for meta-commentary filter in agent loop."""

from __future__ import annotations

import pytest

from godspeed.agent.loop import _strip_meta_commentary


class TestStripMetaCommentary:
    """Test that meta-commentary phrases are stripped from model output."""

    def test_strips_exact_phrase(self) -> None:
        text = "No function call is needed for this prompt."
        assert _strip_meta_commentary(text) == "for this prompt."

    def test_strips_alternative_phrase(self) -> None:
        text = "I don't need any tools. Just answering directly."
        assert _strip_meta_commentary(text) == "Just answering directly."

    def test_no_change_for_normal_text(self) -> None:
        text = "Hello! How can I help you today?"
        assert _strip_meta_commentary(text) == text

    def test_strips_multiple_phrases(self) -> None:
        text = "No function call is needed. No tool call is needed. Hello!"
        assert _strip_meta_commentary(text) == "Hello!"

    def test_empty_after_stripping(self) -> None:
        text = "No function call is needed"
        assert _strip_meta_commentary(text) == ""

    def test_cleans_double_spaces(self) -> None:
        text = "No tool call is needed  for this prompt."
        assert _strip_meta_commentary(text) == "for this prompt."
