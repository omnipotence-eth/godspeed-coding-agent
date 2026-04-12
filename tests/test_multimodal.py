"""Tests for multimodal message support."""

from __future__ import annotations

import pytest

from godspeed.agent.conversation import (
    Conversation,
    build_image_content_block,
    build_multimodal_message,
)
from godspeed.llm.token_counter import IMAGE_BLOCK_TOKEN_ESTIMATE, count_message_tokens


class TestMultimodalMessages:
    """Test multimodal content block support in Conversation."""

    def test_add_user_message_string_unchanged(self) -> None:
        """Plain string messages work exactly as before."""
        conv = Conversation("System prompt")
        conv.add_user_message("Hello world")
        msg = conv.messages[-1]
        assert msg["role"] == "user"
        assert msg["content"] == "Hello world"

    def test_add_user_message_content_blocks(self) -> None:
        """Content block lists are stored as-is."""
        conv = Conversation("System prompt")
        blocks = [
            {"type": "text", "text": "Look at this image"},
            {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
        ]
        conv.add_user_message(blocks)
        msg = conv.messages[-1]
        assert msg["role"] == "user"
        assert isinstance(msg["content"], list)
        assert len(msg["content"]) == 2
        assert msg["content"][0]["type"] == "text"
        assert msg["content"][1]["type"] == "image_url"


class TestBuildImageContentBlock:
    """Test image content block builder."""

    def test_https_url(self) -> None:
        block = build_image_content_block("https://example.com/image.png")
        assert block["type"] == "image_url"
        assert block["image_url"]["url"] == "https://example.com/image.png"

    def test_http_url(self) -> None:
        block = build_image_content_block("http://localhost:8080/img.jpg")
        assert block["type"] == "image_url"
        assert block["image_url"]["url"] == "http://localhost:8080/img.jpg"

    def test_base64_data_uri(self) -> None:
        uri = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=="
        block = build_image_content_block(uri)
        assert block["type"] == "image_url"
        assert block["image_url"]["url"] == uri

    def test_empty_url_raises(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            build_image_content_block("")

    def test_invalid_url_format_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid image URL format"):
            build_image_content_block("/local/path/image.png")

    def test_ftp_url_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid image URL format"):
            build_image_content_block("ftp://example.com/image.png")


class TestBuildMultimodalMessage:
    """Test multimodal message builder."""

    def test_text_only(self) -> None:
        blocks = build_multimodal_message("Hello")
        assert len(blocks) == 1
        assert blocks[0] == {"type": "text", "text": "Hello"}

    def test_text_and_images(self) -> None:
        blocks = build_multimodal_message(
            "Check these images",
            images=["https://example.com/a.png", "https://example.com/b.png"],
        )
        assert len(blocks) == 3
        assert blocks[0]["type"] == "text"
        assert blocks[1]["type"] == "image_url"
        assert blocks[2]["type"] == "image_url"

    def test_images_only(self) -> None:
        blocks = build_multimodal_message("", images=["https://example.com/a.png"])
        assert len(blocks) == 1
        assert blocks[0]["type"] == "image_url"

    def test_no_images_param(self) -> None:
        blocks = build_multimodal_message("Hello", images=None)
        assert len(blocks) == 1
        assert blocks[0]["type"] == "text"

    def test_empty_images_list(self) -> None:
        blocks = build_multimodal_message("Hello", images=[])
        assert len(blocks) == 1


class TestTokenCountWithImages:
    """Test that image content blocks are counted with flat estimate."""

    def test_image_block_counted(self) -> None:
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
                ],
            }
        ]
        count = count_message_tokens(msgs)
        # Should include: 4 (msg overhead) + text tokens + IMAGE_BLOCK_TOKEN_ESTIMATE + 2 (priming)
        assert count >= IMAGE_BLOCK_TOKEN_ESTIMATE

    def test_multiple_images_counted(self) -> None:
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "https://a.com/1.png"}},
                    {"type": "image_url", "image_url": {"url": "https://a.com/2.png"}},
                ],
            }
        ]
        count = count_message_tokens(msgs)
        assert count >= IMAGE_BLOCK_TOKEN_ESTIMATE * 2

    def test_no_images_unchanged(self) -> None:
        """Messages without images should count the same as before."""
        msgs = [{"role": "user", "content": "Hello world"}]
        count = count_message_tokens(msgs)
        # Should be small — just text tokens + overhead
        assert 0 < count < IMAGE_BLOCK_TOKEN_ESTIMATE
