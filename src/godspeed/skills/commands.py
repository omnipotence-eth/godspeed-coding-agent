"""Wire the full skill system into the TUI command layer."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from godspeed.skills.dream import SkillDream
from godspeed.skills.evolution import Lesson, SkillEvolution
from godspeed.skills.loader import Skill, SkillError, SkillHub, SkillSecurityError
from godspeed.skills.security import classify_risk, scan_skill
from godspeed.skills.wiki_bridge import WikiBridge
from godspeed.tui import output as _output
from godspeed.tui.theme import BOLD_PRIMARY, DIM, ERROR, NEUTRAL, SUCCESS, TABLE_BORDER

logger = logging.getLogger(__name__)


def register_skill_commands(
    commands: Any,
    conversation: Any,
    skills: list[Skill],
    evolution: SkillEvolution | None = None,
    hub: SkillHub | None = None,
    dream: SkillDream | None = None,
    skills_dir: Path | None = None,
    llm_client: Any | None = None,
) -> None:
    """Register all skill-related slash commands.

    Called during TUI startup. Injects ``/<trigger>`` for every skill,
    plus management commands: ``/skills``, ``/skill``, ``/skill-evolve``,
    ``/skill-dream``, ``/skill-scan``.
    """
    from godspeed.tui.commands import CommandResult

    evo = evolution or SkillEvolution()
    skill_hub = hub or SkillHub()
    skill_dream = dream or SkillDream()
    sdir = Path(skills_dir) if skills_dir else Path.home() / ".godspeed" / "skills"

    # ── Per-skill activation commands ────────────────────────────────
    for skill in skills:

        def _make_activate(s: Skill, evo_: SkillEvolution) -> Any:
            def _handler(_args: str = "") -> CommandResult:
                content = f"[Skill: {s.name}]\n{s.content}"
                conversation.add_user_message(content)
                evo_.record_lesson(
                    Lesson(
                        skill_name=s.name,
                        text=f"Skill {s.name} was activated",
                        confidence=0.2,
                        source="activation",
                    )
                )
                logger.info("Skill activated name=%s trigger=%s", s.name, s.trigger)
                _output.console.print(
                    f"  [{DIM}]Activated skill:[/{DIM}] [{BOLD_PRIMARY}]{s.name}[/{BOLD_PRIMARY}]"
                )
                return CommandResult(handled=False)

            return _handler

        commands.register(f"/{skill.trigger}", _make_activate(skill, evo))

    # ── /skills — list all ───────────────────────────────────────────
    def _cmd_skills(_args: str = "") -> CommandResult:
        if not skills:
            _output.console.print(f"  [{NEUTRAL}]No skills installed.[/{NEUTRAL}]")
            _output.console.print(
                f"  [{DIM}]Install: /skill install <path>  |  Browse: /skill list[/{DIM}]"
            )
            return CommandResult()

        from rich.table import Table

        table = Table(title="Skills", border_style=TABLE_BORDER, expand=False)
        table.add_column("Trigger", style=BOLD_PRIMARY)
        table.add_column("Name")
        table.add_column("Description", style=NEUTRAL)
        table.add_column("Risk")

        for s in sorted(skills, key=lambda x: x.trigger):
            risk = "clean"
            issues = scan_skill(s.path.parent)
            if issues:
                risk = classify_risk(issues)
            risk_color = SUCCESS if risk == "clean" else ERROR if risk == "dangerous" else "yellow"
            table.add_row(f"/{s.trigger}", s.name, s.description, f"[{risk_color}]{risk}[/]")

        _output.console.print(table)
        return CommandResult()

    commands.register("/skills", _cmd_skills)

    # ── /skill — management sub-commands ─────────────────────────────
    def _cmd_skill(args: str = "") -> CommandResult:
        parts = args.strip().split(maxsplit=1)
        sub = parts[0].lower() if parts else ""
        rest = parts[1] if len(parts) > 1 else ""

        if sub == "list":
            return _cmd_skills()

        elif sub == "install":
            source = Path(rest)
            if not source.is_dir():
                _output.console.print(f"  [{ERROR}]Directory not found: {source}[/{ERROR}]")
                return CommandResult()
            try:
                name = source.name
                skill_hub.install(name, source)
                _output.console.print(f"  [{SUCCESS}]Installed skill: {name}[/{SUCCESS}]")
            except SkillSecurityError as e:
                _output.console.print(f"  [{ERROR}]Security: {e}[/{ERROR}]")
            except SkillError as e:
                _output.console.print(f"  [{ERROR}]{e}[/{ERROR}]")
            return CommandResult()

        elif sub == "remove":
            if not rest:
                _output.console.print(f"  [{ERROR}]Usage: /skill remove <name>[/{ERROR}]")
                return CommandResult()
            try:
                skill_hub.remove(rest)
                _output.console.print(f"  [{SUCCESS}]Removed skill: {rest}[/{SUCCESS}]")
            except SkillError as e:
                _output.console.print(f"  [{ERROR}]{e}[/{ERROR}]")
            return CommandResult()

        elif sub == "scan":
            target = Path(rest) if rest else sdir
            issues = scan_skill(target)
            if not issues:
                _output.console.print(f"  [{SUCCESS}]Skill is clean: {target.name}[/{SUCCESS}]")
            else:
                risk = classify_risk(issues)
                risk_color = ERROR if risk == "dangerous" else "yellow"
                _output.console.print(f"  [{risk_color}]{risk.upper()}:[/]")
                for i in issues:
                    _output.console.print(f"    [{DIM}]- {i}[/{DIM}]")
            return CommandResult()

        elif sub == "verify":
            if not rest:
                _output.console.print(f"  [{ERROR}]Usage: /skill verify <name>[/{ERROR}]")
                return CommandResult()
            ok = skill_hub.verify_integrity(rest)
            if ok:
                _output.console.print(f"  [{SUCCESS}]Integrity OK: {rest}[/{SUCCESS}]")
            else:
                _output.console.print(f"  [{ERROR}]Integrity FAILED: {rest} (tampered?)[/{ERROR}]")
            return CommandResult()

        elif sub == "hub":
            installed = skill_hub.list_installed()
            if not installed:
                _output.console.print(f"  [{NEUTRAL}]No skills in hub.[/{NEUTRAL}]")
                return CommandResult()
            from rich.table import Table

            table = Table(title="Skill Hub", border_style=TABLE_BORDER)
            table.add_column("Name", style=BOLD_PRIMARY)
            table.add_column("Version")
            table.add_column("Installed")
            for entry in installed:
                name = entry["name"]
                version = entry.get("version", "")
                installed_at = entry.get("installed_at", "")
                table.add_row(name, version, installed_at)
            _output.console.print(table)
            return CommandResult()

        elif sub == "generate":
            if not rest:
                msg = f"  [{ERROR}]Usage: /skill generate <topic> [--as <name>][/{ERROR}]"
                _output.console.print(msg)
                return CommandResult()
            parts = rest.split(" --as ", maxsplit=1)
            topic = parts[0].strip()
            output_name = parts[1].strip() if len(parts) > 1 else None
            bridge = WikiBridge()
            result = bridge.generate_skill(topic, output_name=output_name)
            if result:
                _output.console.print(
                    f"  [{SUCCESS}]Generated skill from '{topic}' at: {result}[/{SUCCESS}]"
                )
            else:
                _output.console.print(f"  [{ERROR}]No wiki page found for topic: {topic}[/{ERROR}]")
            return CommandResult()

        else:
            _output.console.print(
                f"  [{NEUTRAL}]Usage: /skill <list|install|remove|scan|verify|hub|generate>"
                f" [args][/{NEUTRAL}]"
            )
            return CommandResult()

    commands.register("/skill", _cmd_skill)

    # ── /skill-evolve — rewrite a skill from accumulated lessons ─────
    def _cmd_skill_evolve(args: str = "") -> CommandResult:
        target_name = args.strip()
        if not target_name:
            _output.console.print(f"  [{ERROR}]Usage: /skill-evolve <name>[/{ERROR}]")
            return CommandResult()

        skill_path = sdir / target_name / "SKILL.md"
        if not skill_path.is_file():
            _output.console.print(f"  [{ERROR}]Skill not found: {target_name}[/{ERROR}]")
            return CommandResult()

        updated = evo.evolve(target_name, skill_path)
        if updated:
            _output.console.print(
                f"  [{SUCCESS}]Evolved skill: {target_name} (backup created)[/{SUCCESS}]"
            )
        else:
            _output.console.print(
                f"  [{NEUTRAL}]No eligible lessons for: {target_name}[/{NEUTRAL}]"
            )
        return CommandResult()

    commands.register("/skill-evolve", _cmd_skill_evolve)

    # ── /skill-dream — manual consolidation trigger ──────────────────
    def _cmd_skill_dream(_args: str = "") -> CommandResult:
        stats = skill_dream.run(sdir)
        _output.console.print(f"  [{SUCCESS}]Dream consolidation:[/{SUCCESS}]")
        _output.console.print(f"    [{DIM}]Dates normalized: {stats['dates_normalized']}[/{DIM}]")
        _output.console.print(f"    [{DIM}]Errors: {stats['errors']}[/{DIM}]")
        return CommandResult()

    commands.register("/skill-dream", _cmd_skill_dream)
