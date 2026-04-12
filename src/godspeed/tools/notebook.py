"""Notebook edit tool — cell-level operations on .ipynb files."""

from __future__ import annotations

import json
import logging
from typing import Any

from godspeed.tools.base import RiskLevel, Tool, ToolContext, ToolResult
from godspeed.tools.path_utils import resolve_tool_path

logger = logging.getLogger(__name__)


class NotebookEditTool(Tool):
    """Edit Jupyter notebooks at the cell level.

    Supports adding, editing, deleting, and moving cells within .ipynb files.
    """

    @property
    def name(self) -> str:
        return "notebook_edit"

    @property
    def description(self) -> str:
        return (
            "Edit Jupyter notebook cells. Actions: edit_cell, add_cell, delete_cell, "
            "move_cell. Operates on .ipynb files by cell index."
        )

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.LOW

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the .ipynb notebook file",
                },
                "action": {
                    "type": "string",
                    "enum": ["edit_cell", "add_cell", "delete_cell", "move_cell"],
                    "description": "The cell operation to perform",
                },
                "cell_index": {
                    "type": "integer",
                    "description": "0-based index of the target cell",
                },
                "content": {
                    "type": "string",
                    "description": "New cell content (for edit_cell and add_cell)",
                },
                "cell_type": {
                    "type": "string",
                    "enum": ["code", "markdown", "raw"],
                    "description": "Cell type for add_cell (default: code)",
                },
                "target_index": {
                    "type": "integer",
                    "description": "Destination index for move_cell",
                },
            },
            "required": ["file_path", "action"],
        }

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        file_path_str = arguments.get("file_path", "")
        if not isinstance(file_path_str, str) or not file_path_str.strip():
            return ToolResult.failure("file_path must be a non-empty string")

        if not file_path_str.endswith(".ipynb"):
            return ToolResult.failure("file_path must be a .ipynb file")

        try:
            file_path = resolve_tool_path(file_path_str, context.cwd)
        except ValueError as exc:
            return ToolResult.failure(str(exc))

        if not file_path.exists():
            return ToolResult.failure(f"File not found: {file_path_str}")

        action = arguments.get("action", "")
        if action not in ("edit_cell", "add_cell", "delete_cell", "move_cell"):
            return ToolResult.failure(f"Invalid action: {action}")

        try:
            notebook = json.loads(file_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            return ToolResult.failure(f"Failed to parse notebook: {exc}")

        cells = notebook.get("cells", [])

        if action == "add_cell":
            return self._add_cell(notebook, cells, arguments, file_path)
        elif action == "edit_cell":
            return self._edit_cell(cells, arguments, file_path, notebook)
        elif action == "delete_cell":
            return self._delete_cell(cells, arguments, file_path, notebook)
        elif action == "move_cell":
            return self._move_cell(cells, arguments, file_path, notebook)

        return ToolResult.failure(f"Unknown action: {action}")

    def _add_cell(
        self,
        notebook: dict[str, Any],
        cells: list[dict[str, Any]],
        arguments: dict[str, Any],
        file_path: Any,
    ) -> ToolResult:
        content = arguments.get("content", "")
        cell_type = arguments.get("cell_type", "code")
        cell_index = arguments.get("cell_index")

        new_cell: dict[str, Any] = {
            "cell_type": cell_type,
            "source": content.splitlines(keepends=True) if content else [],
            "metadata": {},
        }
        if cell_type == "code":
            new_cell["execution_count"] = None
            new_cell["outputs"] = []

        if cell_index is not None and 0 <= cell_index <= len(cells):
            cells.insert(cell_index, new_cell)
            pos = f"at index {cell_index}"
        else:
            cells.append(new_cell)
            pos = f"at index {len(cells) - 1}"

        notebook["cells"] = cells
        file_path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
        return ToolResult.success(f"Added {cell_type} cell {pos}")

    def _edit_cell(
        self,
        cells: list[dict[str, Any]],
        arguments: dict[str, Any],
        file_path: Any,
        notebook: dict[str, Any],
    ) -> ToolResult:
        cell_index = arguments.get("cell_index")
        if cell_index is None:
            return ToolResult.failure("cell_index is required for edit_cell")
        if not (0 <= cell_index < len(cells)):
            return ToolResult.failure(f"cell_index {cell_index} out of range (0-{len(cells) - 1})")

        content = arguments.get("content", "")
        cells[cell_index]["source"] = content.splitlines(keepends=True) if content else []

        file_path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
        return ToolResult.success(f"Updated cell {cell_index}")

    def _delete_cell(
        self,
        cells: list[dict[str, Any]],
        arguments: dict[str, Any],
        file_path: Any,
        notebook: dict[str, Any],
    ) -> ToolResult:
        cell_index = arguments.get("cell_index")
        if cell_index is None:
            return ToolResult.failure("cell_index is required for delete_cell")
        if not (0 <= cell_index < len(cells)):
            return ToolResult.failure(f"cell_index {cell_index} out of range (0-{len(cells) - 1})")

        deleted = cells.pop(cell_index)
        cell_type = deleted.get("cell_type", "unknown")
        notebook["cells"] = cells

        file_path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
        return ToolResult.success(f"Deleted {cell_type} cell at index {cell_index}")

    def _move_cell(
        self,
        cells: list[dict[str, Any]],
        arguments: dict[str, Any],
        file_path: Any,
        notebook: dict[str, Any],
    ) -> ToolResult:
        cell_index = arguments.get("cell_index")
        target_index = arguments.get("target_index")
        if cell_index is None or target_index is None:
            return ToolResult.failure("cell_index and target_index are required for move_cell")
        if not (0 <= cell_index < len(cells)):
            return ToolResult.failure(f"cell_index {cell_index} out of range (0-{len(cells) - 1})")
        if not (0 <= target_index < len(cells)):
            return ToolResult.failure(
                f"target_index {target_index} out of range (0-{len(cells) - 1})"
            )

        cell = cells.pop(cell_index)
        cells.insert(target_index, cell)
        notebook["cells"] = cells

        file_path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
        return ToolResult.success(f"Moved cell from {cell_index} to {target_index}")
