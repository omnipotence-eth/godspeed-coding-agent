"""Project scaffolding tool - create new projects from templates."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from godspeed.tools.base import RiskLevel, Tool, ToolContext, ToolResult

logger = logging.getLogger(__name__)


class ProjectScaffoldTool(Tool):
    """Create new projects from templates."""

    produces_diff = False

    @property
    def name(self) -> str:
        return "project_scaffold"

    @property
    def description(self) -> str:
        return (
            "Create a new project from templates. "
            "Supports various frameworks and project types."
        )

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.HIGH

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "project_name": {"type": "string"},
                "template": {
                    "type": "string",
                    "enum": ["python", "react", "node", "fastapi", "list"],
                },
                "options": {"type": "object"},
            },
            "required": ["project_name", "template"],
        }

    TEMPLATES = {
        "python": {
            "files": {
                "README.md": "# {name}\n\nA Python project.",
                "pyproject.toml": '[project]\nname = "{name}"\nversion = "0.1.0"',
                "src/__init__.py": "",
                "tests/__init__.py": "",
            },
        },
        "fastapi": {
            "files": {
                "README.md": "# {name}\n\nA FastAPI project.",
                "pyproject.toml": '[project]\nname = "{name}"\nversion = "0.1.0"\n[dependencies]\nfastapi = ">=0.100"',
                "src/__init__.py": "",
                "src/main.py": "from fastapi import FastAPI\napp = FastAPI()\n@app.get(\"/\")\ndef read_root():\n    return {'Hello': 'World'}",
            },
        },
        "react": {
            "files": {
                "README.md": "# {name}\n\nA React project.",
                "package.json": '{{"name": "{name}", "version": "0.0.0", "type": "module", "scripts": {{"dev": "vite"}}}}',
                "index.html": "<!DOCTYPE html>\n<html><body><div id='root'></div><script type='module' src='/src/main.jsx'></script></body></html>",
                "src/main.jsx": "import React from 'react'\nimport ReactDOM from 'react-dom/client'\nimport App from './App.jsx'\nReactDOM.createRoot(document.getElementById('root')).render(<App />)",
                "src/App.jsx": "export default function App() {{ return <div>Hello World</div> }}",
            },
        },
        "node": {
            "files": {
                "README.md": "# {name}\n\nA Node.js project.",
                "package.json": '{{"name": "{name}", "version": "0.0.0", "main": "index.js"}}',
                "index.js": "console.log('Hello world!')",
            },
        },
    }

    async def execute(
        self,
        tool_context: ToolContext,
        project_name: str,
        template: str = "python",
        options: dict[str, Any] | None = None,
    ) -> ToolResult:
        """Create a new project from template."""
        if template == "list":
            lines = ["## Templates\n"]
            for name, info in self.TEMPLATES.items():
                lines.append(f"- **{name}**")
            return ToolResult.ok("\n".join(lines))

        if template not in self.TEMPLATES:
            return ToolResult.failure(f"Unknown template: {template}")

        project_dir = tool_context.cwd / project_name

        if project_dir.exists():
            return ToolResult.failure(f"Project exists: {project_name}")

        try:
            project_dir.mkdir(parents=True)
        except Exception as exc:
            return ToolResult.failure(f"Cannot create: {exc}")

        files_created = []
        for file_path, content in self.TEMPLATES[template]["files"].items():
            full_path = project_dir / file_path.replace("{name}", project_name)
            full_path.parent.mkdir(parents=True, exist_ok=True)
            with open(full_path, "w") as f:
                f.write(content.replace("{name}", project_name))
            files_created.append(file_path)

        with open(project_dir / ".gitignore", "w") as f:
            f.write("node_modules/\n__pycache__/\n*.pyc\n.env\n")

        return ToolResult.ok(f"Created {project_name} with {len(files_created)} files")