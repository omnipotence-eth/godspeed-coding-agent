"""Skill discovery and loading from markdown files with YAML frontmatter."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SkillDefinition:
    """A skill loaded from a markdown file."""

    name: str
    description: str
    trigger: str
    content: str


def _parse_skill_file(path: Path) -> SkillDefinition | None:
    """Parse a single .md skill file with YAML frontmatter.

    Expected format:
        ---
        name: code-review
        description: Review code for best practices
        trigger: review
        ---
        Review the following code for...

    Returns None if the file is malformed or missing required fields.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("Cannot read skill file %s: %s", path, exc)
        return None

    # Parse YAML frontmatter between --- markers
    stripped = text.strip()
    if not stripped.startswith("---"):
        logger.warning("Skill file %s missing YAML frontmatter (no leading ---)", path)
        return None

    # Find the closing --- marker
    second_marker = stripped.find("---", 3)
    if second_marker == -1:
        logger.warning("Skill file %s missing closing --- marker", path)
        return None

    frontmatter_str = stripped[3:second_marker].strip()
    body = stripped[second_marker + 3 :].strip()

    try:
        frontmatter = yaml.safe_load(frontmatter_str)
    except yaml.YAMLError as exc:
        logger.warning("Invalid YAML frontmatter in %s: %s", path, exc)
        return None

    if not isinstance(frontmatter, dict):
        logger.warning("Skill file %s frontmatter is not a mapping", path)
        return None

    # Validate required fields
    name = frontmatter.get("name")
    description = frontmatter.get("description")
    trigger = frontmatter.get("trigger")

    if not all([name, description, trigger]):
        missing = [f for f in ("name", "description", "trigger") if not frontmatter.get(f)]
        logger.warning("Skill file %s missing required fields: %s", path, ", ".join(missing))
        return None

    if not body:
        logger.warning("Skill file %s has empty content body", path)
        return None

    return SkillDefinition(
        name=str(name),
        description=str(description),
        trigger=str(trigger),
        content=body,
    )


def discover_skills(dirs: list[Path]) -> list[SkillDefinition]:
    """Discover skills from directories, later dirs override earlier on same trigger.

    Scan order: global (~/.godspeed/skills/) then project (.godspeed/skills/).
    Project skills override global skills with the same trigger name.

    Args:
        dirs: Directories to scan for .md skill files.

    Returns:
        List of SkillDefinition objects, deduplicated by trigger.
    """
    skills_by_trigger: dict[str, SkillDefinition] = {}

    for directory in dirs:
        if not directory.is_dir():
            continue

        for md_file in sorted(directory.glob("*.md")):
            skill = _parse_skill_file(md_file)
            if skill is not None:
                if skill.trigger in skills_by_trigger:
                    logger.info(
                        "Skill trigger '%s' overridden by %s",
                        skill.trigger,
                        md_file,
                    )
                skills_by_trigger[skill.trigger] = skill

    logger.info("Discovered %d skills from %d directories", len(skills_by_trigger), len(dirs))
    return list(skills_by_trigger.values())
