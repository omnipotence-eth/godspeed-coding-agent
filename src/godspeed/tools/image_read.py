"""Image read tool — read image files as base64-encoded content for vision LLMs."""

from __future__ import annotations

import base64
import logging
from typing import Any

from godspeed.tools.base import RiskLevel, Tool, ToolContext, ToolResult
from godspeed.tools.path_utils import resolve_tool_path

logger = logging.getLogger(__name__)

MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024  # 20MB hard limit
WARN_FILE_SIZE_BYTES = 5 * 1024 * 1024  # 5MB warning threshold

SUPPORTED_FORMATS: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


class ImageReadTool(Tool):
    """Read an image file and return base64-encoded content for vision-capable LLMs.

    Supports PNG, JPG/JPEG, GIF, and WebP formats. Returns a JSON object
    with a data URI suitable for multimodal LLM content blocks.
    """

    @property
    def name(self) -> str:
        return "image_read"

    @property
    def description(self) -> str:
        return (
            "Read an image file from disk and return base64-encoded content "
            "for vision-capable LLMs. Supports PNG, JPG, JPEG, GIF, and WebP."
        )

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.READ_ONLY

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the image file to read (relative to project root)",
                },
            },
            "required": ["file_path"],
        }

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        file_path_str = arguments.get("file_path", "")
        if not isinstance(file_path_str, str) or not file_path_str:
            return ToolResult.failure("file_path must be a non-empty string")

        try:
            file_path = resolve_tool_path(file_path_str, context.cwd)
        except ValueError as exc:
            return ToolResult.failure(str(exc))

        if not file_path.exists():
            return ToolResult.failure(f"File not found: {file_path}")

        if not file_path.is_file():
            return ToolResult.failure(f"Not a file (is a directory?): {file_path}")

        # Check supported format
        suffix = file_path.suffix.lower()
        mime_type = SUPPORTED_FORMATS.get(suffix)
        if mime_type is None:
            supported = ", ".join(sorted(SUPPORTED_FORMATS))
            return ToolResult.failure(
                f"Unsupported image format '{suffix}'. Supported formats: {supported}"
            )

        # Size checks
        file_size = file_path.stat().st_size
        if file_size > MAX_FILE_SIZE_BYTES:
            size_mb = file_size / (1024 * 1024)
            return ToolResult.failure(f"Image too large: {size_mb:.1f}MB (max 20MB)")

        warning = ""
        if file_size > WARN_FILE_SIZE_BYTES:
            size_mb = file_size / (1024 * 1024)
            warning = f" [WARNING: large image {size_mb:.1f}MB]"
            logger.warning("large image file size=%d path=%s", file_size, file_path)

        # Read and encode
        try:
            raw_bytes = file_path.read_bytes()
        except OSError as exc:
            return ToolResult.failure(f"Failed to read image: {exc}")

        b64_data = base64.b64encode(raw_bytes).decode("ascii")
        size_kb = file_size / 1024
        filename = file_path.name

        output = (
            f"[Image: {filename}] ({mime_type}, {size_kb:.1f}KB){warning}\n"
            f"data:{mime_type};base64,{b64_data}"
        )

        return ToolResult.success(output)
