"""API Documentation generator."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from godspeed.tools.base import RiskLevel, Tool, ToolContext, ToolResult

logger = logging.getLogger(__name__)


class DocGeneratorTool(Tool):
    """Generate API documentation from code.

    Creates documentation from docstrings, type hints,
    and code structure.
    """

    produces_diff = False

    @property
    def name(self) -> str:
        return "generate_docs"

    @property
    def description(self) -> str:
        return (
            "Generate API documentation from Python code. "
            "Creates docs from docstrings and type hints."
        )

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.LOW

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "File or directory to document"},
                "output": {"type": "string", "description": "Output file path"},
                "format": {
                    "type": "string",
                    "enum": ["markdown", "html", "restructuredtext"],
                    "description": "Output format",
                },
            },
            "required": ["file_path"],
        }

    async def execute(
        self,
        tool_context: ToolContext,
        file_path: str,
        output: str | None = None,
        format: str = "markdown",
    ) -> ToolResult:
        """Generate documentation."""
        import os

        if not os.path.exists(file_path):
            return ToolResult.failure(f"Not found: {file_path}")

        is_dir = os.path.isdir(file_path)
        files = []

        if is_dir:
            for root, _, filenames in os.walk(file_path):
                for fn in filenames:
                    if fn.endswith(".py"):
                        files.append(os.path.join(root, fn))
        else:
            files = [file_path]

        all_docs = []

        for fp in files:
            docs = self._generate_docs_for_file(fp, format)
            all_docs.extend(docs)

        doc_output = "\n\n".join(all_docs)

        if output:
            with open(output, "w") as f:
                f.write(doc_output)
            return ToolResult.ok(f"Documentation written to: {output}")

        return ToolResult.ok(doc_output)

    def _generate_docs_for_file(self, file_path: str, format: str) -> list[str]:
        """Generate docs for a single file."""
        import ast

        docs = []

        try:
            with open(file_path) as f:
                tree = ast.parse(f.read())
        except Exception:
            return docs

        module_doc = ast.get_docstring(tree)
        if module_doc:
            docs.append(f"# {Path(file_path).stem}\n\n{module_doc}")

        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                doc = ast.get_docstring(node)
                if doc:
                    if isinstance(node, ast.ClassDef):
                        docs.append(f"## class {node.name}\n\n{doc}")
                    else:
                        docs.append(f"### {node.name}()\n\n{doc}")

        return docs


class OpenAPITool(Tool):
    """Generate OpenAPI specification."""

    produces_diff = False

    @property
    def name(self) -> str:
        return "generate_openapi"

    @property
    def description(self) -> str:
        return "Generate OpenAPI spec from FastAPI/Starlette code."

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.LOW

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "output": {"type": "string"},
            },
            "required": ["file_path"],
        }

    async def execute(
        self,
        tool_context: ToolContext,
        file_path: str,
        output: str | None = None,
    ) -> ToolResult:
        """Generate OpenAPI spec."""
        import json
        import os

        if not os.path.exists(file_path):
            return ToolResult.failure(f"Not found: {file_path}")

        try:
            # Try to import the app
            import importlib.util

            spec = importlib.util.spec_from_file_location("app", file_path)
            if not spec or not spec.loader:
                return ToolResult.failure("Cannot load module")

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Find the app
            app = None
            for name in dir(module):
                obj = getattr(module, name)
                if hasattr(obj, "routes"):
                    app = obj
                    break

            if not app:
                return ToolResult.failure("No FastAPI/Starlette app found")

            # Generate spec
            try:
                from fastapi.openapi import to_json
                spec_json = json.loads(to_json(app))
            except ImportError:
                return ToolResult.failure("Install fastapi: pip install fastapi")
            except Exception:
                try:
                    spec_json = app.openapi()
                except Exception as e:
                    return ToolResult.failure(f"Cannot generate: {e}")

            output_json = json.dumps(spec_json, indent=2)

            if output:
                with open(output, "w") as f:
                    f.write(output_json)
                return ToolResult.ok(f"OpenAPI spec written to: {output}")

            return ToolResult.ok(output_json[:1000] + "...")

        except Exception as exc:
            return ToolResult.failure(f"Failed: {exc}")
