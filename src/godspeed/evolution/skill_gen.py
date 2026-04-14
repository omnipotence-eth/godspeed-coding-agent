"""Skill auto-generation — detect repeated tool patterns and create skills.

Analyzes multi-tool sequences from trace data and generates reusable
skill markdown files with YAML frontmatter.
"""

from __future__ import annotations

import dataclasses
import logging
import re

import yaml

from godspeed.evolution.trace_analyzer import SessionTrace, ToolSequence, TraceAnalyzer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class GeneratedSkill:
    """A skill auto-generated from repeated tool patterns."""

    name: str
    description: str
    trigger: str
    content: str  # Full markdown with frontmatter
    source_tools: tuple[str, ...]
    frequency: int


# ---------------------------------------------------------------------------
# Skill Generator
# ---------------------------------------------------------------------------


class SkillGenerator:
    """Detect repeated tool patterns and generate reusable skills."""

    def __init__(self) -> None:
        self._analyzer = TraceAnalyzer()

    def detect_patterns(
        self,
        sessions: list[SessionTrace],
        min_frequency: int = 3,
    ) -> list[ToolSequence]:
        """Find tool chains that repeat ≥min_frequency times."""
        return self._analyzer.analyze_multi_tool_sequences(sessions, min_frequency)

    def generate_skill_markdown(
        self,
        sequence: ToolSequence,
        description: str | None = None,
    ) -> str:
        """Produce a skill file with YAML frontmatter from a tool sequence.

        Args:
            sequence: The detected tool pattern.
            description: Override description. Auto-generated if None.

        Returns:
            Complete skill markdown with frontmatter.
        """
        name = self._make_name(sequence.tools)
        trigger = name  # Use name as trigger
        desc = description or self._make_description(sequence.tools)

        frontmatter = {
            "name": name,
            "description": desc,
            "trigger": trigger,
        }

        steps = []
        for i, tool in enumerate(sequence.tools, 1):
            steps.append(f"{i}. Use `{tool}` tool")

        body = "\n".join(steps)
        header = yaml.dump(frontmatter, default_flow_style=False).strip()
        skill_text = f"---\n{header}\n---\n\n{body}\n"
        return skill_text

    def generate(
        self,
        sequence: ToolSequence,
        description: str | None = None,
    ) -> GeneratedSkill:
        """Generate a full GeneratedSkill from a sequence."""
        name = self._make_name(sequence.tools)
        desc = description or self._make_description(sequence.tools)
        content = self.generate_skill_markdown(sequence, description)

        return GeneratedSkill(
            name=name,
            description=desc,
            trigger=name,
            content=content,
            source_tools=sequence.tools,
            frequency=sequence.frequency,
        )

    def validate_skill(self, skill_text: str) -> bool:
        """Validate that a skill has proper YAML frontmatter and required fields."""
        parts = skill_text.split("---")
        if len(parts) < 3:
            return False

        try:
            metadata = yaml.safe_load(parts[1])
        except yaml.YAMLError:
            return False

        if not isinstance(metadata, dict):
            return False

        required = {"name", "description", "trigger"}
        if not required.issubset(metadata.keys()):
            return False

        # Check body has content
        body = "---".join(parts[2:]).strip()
        return len(body) > 0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_name(tools: tuple[str, ...]) -> str:
        """Generate a kebab-case name from tool names."""
        # Deduplicate consecutive tools
        deduped: list[str] = []
        for t in tools:
            if not deduped or deduped[-1] != t:
                deduped.append(t)

        name = "-and-".join(t.replace("_", "-") for t in deduped)
        # Sanitize
        name = re.sub(r"[^a-z0-9-]", "", name.lower())
        return name[:50]  # Cap length

    @staticmethod
    def _make_description(tools: tuple[str, ...]) -> str:
        """Generate a description from tool names."""
        tool_names = " then ".join(t.replace("_", " ") for t in dict.fromkeys(tools))
        return f"Auto-generated: {tool_names}"
