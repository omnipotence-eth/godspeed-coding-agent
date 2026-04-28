"""Wire skill definitions into the TUI command system."""

from __future__ import annotations

import logging
from typing import Any

from godspeed.skills.loader import SkillDefinition
from godspeed.tui import output as _output
from godspeed.tui.theme import BOLD_PRIMARY, DIM, NEUTRAL, TABLE_BORDER

logger = logging.getLogger(__name__)


def register_skill_commands(
    commands: Any,
    conversation: Any,
    skills: list[SkillDefinition],
) -> None:
    """Register slash commands for each discovered skill.

    For each skill, registers ``/{trigger}`` that injects the skill content
    as a user message and returns ``CommandResult(handled=False)`` so the
    injected content flows through to ``agent_loop()``.

    Also registers ``/skills`` to list available skills.

    Args:
        commands: Commands instance (duck-typed to avoid circular import).
        conversation: Conversation instance for message injection.
        skills: List of discovered skill definitions.
    """
    from godspeed.tui.commands import CommandResult

    for skill in skills:

        def _make_handler(s: SkillDefinition) -> Any:
            def _handler(_args: str = "") -> CommandResult:
                # Inject skill content as user message
                content = f"[Skill: {s.name}]\n{s.content}"
                conversation.add_user_message(content)
                logger.info("Skill activated name=%s trigger=%s", s.name, s.trigger)
                _output.console.print(
                    f"  [{DIM}]Activated skill:[/{DIM}] [{BOLD_PRIMARY}]{s.name}[/{BOLD_PRIMARY}]"
                )
                # handled=False so the TUI runs agent_loop with the injected message
                return CommandResult(handled=False)

            return _handler

        commands.register(f"/{skill.trigger}", _make_handler(skill))
        logger.debug("Registered skill command /%s → %s", skill.trigger, skill.name)

    # Register /skills listing command
    def _cmd_skills(_args: str = "") -> CommandResult:
        if not skills:
            _output.console.print(f"  [{NEUTRAL}]No skills installed.[/{NEUTRAL}]")
            _output.console.print(
                f"  [{DIM}]Add .md files to ~/.godspeed/skills/ or .godspeed/skills/[/{DIM}]"
            )
            return CommandResult()

        from rich.table import Table

        table = Table(title="Skills", border_style=TABLE_BORDER, expand=False)
        table.add_column("Trigger", style=BOLD_PRIMARY)
        table.add_column("Name")
        table.add_column("Description", style=NEUTRAL)

        for s in sorted(skills, key=lambda x: x.trigger):
            table.add_row(f"/{s.trigger}", s.name, s.description)

        _output.console.print(table)
        return CommandResult()

    commands.register("/skills", _cmd_skills)
