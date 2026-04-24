"""Ask clarification tool — pause and ask user to choose from options.

Inspired by Claude Code's askUserQuestion capability. This allows the agent
to pause mid-task and ask the user to clarify or choose from options,
making the workflow collaborative rather than purely autonomous.
"""

from __future__ import annotations

import logging
from typing import Any

from godspeed.tools.base import RiskLevel, Tool, ToolContext, ToolResult

logger = logging.getLogger(__name__)


class AskClarificationTool(Tool):
    """Ask the user a question and wait for their response.

    The agent pauses and presents options to the user. This makes the workflow
    collaborative — the agent can ask for clarification when stuck or when
    multiple valid approaches exist.

    Use cases:
    - Ambiguous task: "Should I refactor X or Y?"
    - Risky operation: "Delete 10 files? Confirm:"
    - Multiple approaches: "Use strategy A, B, or C?"
    - Missing information: "What should X be?"
    """

    produces_diff = False

    @property
    def name(self) -> str:
        return "ask_clarification"

    @property
    def description(self) -> str:
        return (
            "Ask the user a question and wait for a response. Use when you need "
            "clarification, confirmation, or want to present options. The agent pauses "
            "until the user responds.\n\n"
            "Example: ask_clarification(question='Should I delete the old files?', "
            "options=['Yes, delete all', 'No, keep them', 'Ask for each file'])\n"
            "Example: ask_clarification(question='What naming convention?', "
            "options=['snake_case', 'camelCase', 'PascalCase'])"
        )

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.READ_ONLY  # No destructive action, just asking

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question to ask the user",
                    "examples": [
                        "Should I refactor the authentication module or the API layer first?",
                        "Do you want me to delete the test files?",
                    ],
                },
                "options": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of options for the user to choose from (2-5 options recommended)",
                    "minItems": 1,
                    "maxItems": 5,
                    "examples": [
                        ["Refactor auth first", "Refactor API first", "Do both"],
                        ["Yes", "No", "Ask for each"],
                    ],
                },
            },
            "required": ["question", "options"],
        }

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        question = arguments.get("question", "")
        options = arguments.get("options", [])

        if not question:
            return ToolResult.failure("question is required")
        if not options:
            return ToolResult.failure("options must have at least one choice")
        if len(options) > 5:
            return ToolResult.failure("options cannot exceed 5 choices")

        logger.info("Agent asking clarification: %s", question)

        if context.ask_user_callback:
            try:
                response = await context.ask_user_callback(question, options)
                return ToolResult.success(
                    f"User responded: {response}",
                    extra={"user_response": response, "question": question, "options": options},
                )
            except Exception as e:
                return ToolResult.failure(f"Failed to ask user: {e}")

        return ToolResult.success(
            f"[WAITING FOR USER INPUT]\nQuestion: {question}\nOptions: {options}\n"
            f"Please respond with one of: {', '.join(options)}",
            extra={"awaiting_response": True, "question": question, "options": options},
        )
