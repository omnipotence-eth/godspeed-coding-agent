"""Workflow automation for repetitive trajectories."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from godspeed.tools.base import RiskLevel, Tool, ToolContext, ToolResult

logger = logging.getLogger(__name__)


class WorkflowTool(Tool):
    """Manage and execute reusable workflows."""

    produces_diff = False

    @property
    def name(self) -> str:
        return "workflow"

    @property
    def description(self) -> str:
        return (
            "Create, list, and run reusable workflows. "
            "Workflows automate repetitive tasks."
        )

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.HIGH

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list", "run", "save"],
                    "description": "Action to perform",
                },
                "workflow_name": {
                    "type": "string",
                    "description": "Name of workflow",
                },
                "description": {
                    "type": "string",
                    "description": "Workflow description",
                },
                "steps": {
                    "type": "array",
                    "description": "Workflow steps",
                },
            },
            "required": ["action"],
        }

    async def execute(
        self,
        tool_context: ToolContext,
        action: str,
        workflow_name: str | None = None,
        description: str | None = None,
        steps: list[dict[str, Any]] | None = None,
    ) -> ToolResult:
        """Execute workflow action."""
        store = WorkflowStore()

        if action == "list":
            workflows = store.list_workflows()
            if not workflows:
                return ToolResult.ok("No workflows saved")

            lines = ["## Workflows"]
            for w in workflows:
                lines.append(f"- {w['name']}: {w['description']} ({w['steps']} steps)")

            return ToolResult.ok("\n".join(lines))

        elif action == "run":
            if not workflow_name:
                return ToolResult.failure("workflow_name required for run")

            workflow = store.load_workflow(workflow_name)
            if not workflow:
                return ToolResult.failure(f"Workflow not found: {workflow_name}")

            results = []
            for step in workflow.steps:
                tool_name = step.get("tool")
                tool_args = step.get("args", {})
                result = await tool_context.tool_registry.dispatch(tool_name, tool_args)
                results.append({"tool": tool_name, "result": result})

            return ToolResult.ok(f"Workflow '{workflow_name}' completed with {len(results)} steps")

        elif action == "save":
            if not workflow_name or not steps:
                return ToolResult.failure("workflow_name and steps required for save")

            workflow = Workflow(workflow_name, description or "", steps)
            store.save_workflow(workflow)
            return ToolResult.ok(f"Saved workflow: {workflow_name}")

        return ToolResult.failure(f"Unknown action: {action}")


class Workflow:
    """A reusable workflow."""

    def __init__(
        self,
        name: str,
        description: str,
        steps: list[dict[str, Any]],
    ) -> None:
        self.name = name
        self.description = description
        self.steps = steps


class WorkflowStore:
    """Store and manage workflows."""

    def __init__(self) -> None:

        self.db_path = Path.home() / ".godspeed" / "workflows"
        self.db_path.mkdir(parents=True, exist_ok=True)

    def save_workflow(self, workflow: Workflow) -> None:
        """Save a workflow."""
        import json

        with open(self.db_path / f"{workflow.name}.json", "w") as f:
            json.dump(
                {
                    "name": workflow.name,
                    "description": workflow.description,
                    "steps": workflow.steps,
                },
                f,
            )

    def load_workflow(self, name: str) -> Workflow | None:
        """Load a workflow."""
        import json

        path = self.db_path / f"{name}.json"
        if not path.exists():
            return None

        with open(path) as f:
            data = json.load(f)
            return Workflow(
                name=data["name"],
                description=data["description"],
                steps=data["steps"],
            )

    def list_workflows(self) -> list[dict[str, Any]]:
        """List workflows."""
        import json

        workflows = []
        for path in self.db_path.glob("*.json"):
            with open(path) as f:
                data = json.load(f)
                workflows.append(
                    {
                        "name": data["name"],
                        "description": data["description"],
                        "steps": len(data["steps"]),
                    }
                )
        return workflows
