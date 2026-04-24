"""Multi-file batch edit tool — apply search/replace to multiple files."""

from __future__ import annotations

import logging
from typing import Any

from godspeed.tools.base import RiskLevel, Tool, ToolContext, ToolResult

logger = logging.getLogger(__name__)


class BatchEditTool(Tool):
    """Apply the same search/replace pattern to multiple files.

    Useful for refactoring across many files (e.g., renaming a function
    everywhere, updating import paths, etc.).
    """

    produces_diff = True

    @property
    def name(self) -> str:
        return "batch_edit"

    @property
    def description(self) -> str:
        return (
            "Apply a search/replace pattern to multiple files at once. "
            "Glob patterns are supported. Returns results for each file."
        )

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.HIGH

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of file paths or glob patterns to edit",
                },
                "old_string": {
                    "type": "string",
                    "description": "The text to find and replace (same for all files)",
                },
                "new_string": {
                    "type": "string",
                    "description": "The replacement text",
                },
                "dry_run": {
                    "type": "boolean",
                    "description": "If true, show what would change without writing",
                },
            },
            "required": ["files", "old_string", "new_string"],
        }

    async def execute(
        self,
        tool_context: ToolContext,
        files: list[str],
        old_string: str,
        new_string: str,
        dry_run: bool = False,
    ) -> ToolResult:
        """Apply the same edit to all matching files."""
        import glob
        import os

        from godspeed.tools.file_edit import FileEditTool

        all_paths: set[str] = set()
        for pattern in files:
            if "*" in pattern or "?" in pattern:
                matches = glob.glob(pattern, recursive=True)
                all_paths.update(matches)
            else:
                all_paths.add(pattern)

        if not all_paths:
            return ToolResult(
                ok=False,
                error=f"No files found matching: {files}",
            )

        results: list[dict[str, Any]] = []
        edit_tool = FileEditTool()

        for path in sorted(all_paths):
            if not os.path.isfile(path):
                results.append({"file": path, "status": "skipped", "reason": "not a file"})
                continue

            try:
                result = await edit_tool.execute(
                    tool_context,
                    file_path=path,
                    old_string=old_string,
                    new_string=new_string,
                )
                results.append({
                    "file": path,
                    "status": "applied" if result.ok else "error",
                    "error": result.error,
                    "output": result.output,
                })
            except Exception as exc:
                results.append({"file": path, "status": "error", "error": str(exc)})

        success_count = sum(1 for r in results if r.get("status") == "applied")
        error_count = sum(1 for r in results if r.get("status") == "error")

        summary = f"Edited {success_count} of {len(results)} files"
        if error_count:
            summary += f" ({error_count} errors)"

        return ToolResult(
            ok=error_count == 0,
            output=summary,
            results=results,
        )
