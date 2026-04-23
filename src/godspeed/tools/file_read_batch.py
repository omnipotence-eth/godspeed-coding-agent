"""Batch file read tool — read multiple files in one call."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from godspeed.tools.base import RiskLevel, Tool, ToolContext, ToolResult

logger = logging.getLogger(__name__)

MAX_FILES_PER_BATCH = 10
MAX_TOTAL_BYTES = 500_000  # 500KB total limit


class FileReadBatchTool(Tool):
    """Read multiple files in a single call.

    More efficient than individual file_read calls when reading
    multiple files. Returns combined output with file separators.
    """

    @property
    def name(self) -> str:
        return "file_read_batch"

    @property
    def description(self) -> str:
        return (
            "Read multiple files in one call. More efficient than individual "
            f"file_read calls. Max {MAX_FILES_PER_BATCH} files, {MAX_TOTAL_BYTES} bytes total. "
            "Returns combined output with file separators.\n\n"
            "Example: file_read_batch(file_paths=['src/a.py', 'src/b.py'])"
        )

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.READ_ONLY

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": f"List of file paths to read (max {MAX_FILES_PER_BATCH})",
                },
            },
            "required": ["file_paths"],
        }

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        file_paths = arguments.get("file_paths", [])

        if not isinstance(file_paths, list) or not file_paths:
            return ToolResult.failure("file_paths must be a non-empty list")

        if len(file_paths) > MAX_FILES_PER_BATCH:
            return ToolResult.failure(
                f"Too many files: {len(file_paths)} > {MAX_FILES_PER_BATCH} limit"
            )

        results: list[str] = []
        total_bytes = 0
        errors: list[str] = []

        for i, file_path_str in enumerate(file_paths):
            if not isinstance(file_path_str, str) or not file_path_str:
                errors.append(f"[{i}] Invalid path: must be non-empty string")
                continue

            # Resolve path
            try:
                file_path = Path(file_path_str)
                if not file_path.is_absolute():
                    file_path = (context.project_dir or Path.cwd()) / file_path_str
                    file_path = file_path.resolve()
            except Exception as exc:
                errors.append(f"[{i}] Path resolution failed: {exc}")
                continue

            # Security check
            try:
                file_path.absolute().resolve().relative_to(
                    (context.project_dir or Path.cwd()).absolute().resolve()
                )
            except ValueError:
                errors.append(f"[{i}] Path outside project: {file_path_str}")
                continue

            if not file_path.exists():
                errors.append(f"[{i}] File not found: {file_path_str}")
                continue

            if file_path.is_dir():
                errors.append(f"[{i}] Is a directory: {file_path_str}")
                continue

            # Read file
            try:
                content = file_path.read_text(encoding="utf-8")
                if total_bytes + len(content.encode("utf-8")) > MAX_TOTAL_BYTES:
                    errors.append(f"[{i}] Would exceed {MAX_TOTAL_BYTES} byte limit")
                    break

                results.append(f"=== {file_path_str} ===\n{content}")
                total_bytes += len(content.encode("utf-8"))

            except UnicodeDecodeError:
                errors.append(f"[{i}] Binary file (not readable as text): {file_path_str}")
            except OSError as exc:
                errors.append(f"[{i}] Read error: {exc}")

        # Build output
        output_parts = []
        if results:
            output_parts.append("\n\n".join(results))
        if errors:
            output_parts.append("Errors:\n" + "\n".join(errors))

        if not output_parts:
            return ToolResult.success("No files read (all failed or empty)")

        return ToolResult.success("\n\n---\n\n".join(output_parts))
