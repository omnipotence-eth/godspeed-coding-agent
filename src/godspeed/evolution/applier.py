"""Evolution applier — runtime hot-swap of evolved artifacts.

Applies mutations to tool descriptions, system prompt sections, and skills
at runtime. Supports rollback on regression. Loads applied overrides on startup.
"""

from __future__ import annotations

import logging
from pathlib import Path

from godspeed.evolution.registry import EvolutionRegistry

logger = logging.getLogger(__name__)


class EvolutionApplier:
    """Apply evolved artifacts at runtime without restart.

    Manages the lifecycle of applied mutations:
    - Apply: update tool descriptions or save prompt overrides
    - Rollback: restore originals
    - Load: restore applied state on startup
    """

    def __init__(self, registry: EvolutionRegistry) -> None:
        self._registry = registry
        self._description_overrides: dict[str, str] = {}  # tool_name -> description
        self._prompt_overrides: dict[str, str] = {}  # section_name -> text

    @property
    def description_overrides(self) -> dict[str, str]:
        """Currently active tool description overrides."""
        return dict(self._description_overrides)

    @property
    def prompt_overrides(self) -> dict[str, str]:
        """Currently active prompt section overrides."""
        return dict(self._prompt_overrides)

    def load_applied(self) -> int:
        """Load all previously applied mutations from disk.

        Called on startup to restore evolution state.

        Returns:
            Number of overrides loaded.
        """
        count = 0
        for applied_data in self._registry.list_applied():
            artifact_type = applied_data.get("artifact_type", "")
            artifact_id = applied_data.get("artifact_id", "")
            mutated_text = applied_data.get("mutated_text", "")

            if not artifact_id or not mutated_text:
                continue

            if artifact_type == "tool_description":
                self._description_overrides[artifact_id] = mutated_text
                count += 1
            elif artifact_type == "prompt_section":
                self._prompt_overrides[artifact_id] = mutated_text
                count += 1
            elif artifact_type == "skill":
                # Skills are written as files — they persist on disk
                count += 1

        if count > 0:
            logger.info("Loaded %d evolution overrides", count)
        return count

    def apply_tool_description(
        self, record_id: str, tool_name: str, mutated_text: str, original_text: str
    ) -> None:
        """Apply a tool description mutation.

        Updates the in-memory override and persists to registry.
        """
        self._registry.save_original(tool_name, original_text)
        self._registry.mark_applied(record_id, mutated_text)
        self._description_overrides[tool_name] = mutated_text
        logger.info("Applied tool description evolution tool=%s record=%s", tool_name, record_id)

    def apply_prompt_section(
        self, record_id: str, section_name: str, mutated_text: str, original_text: str
    ) -> None:
        """Apply a prompt section mutation."""
        self._registry.save_original(section_name, original_text)
        self._registry.mark_applied(record_id, mutated_text)
        self._prompt_overrides[section_name] = mutated_text
        logger.info(
            "Applied prompt section evolution section=%s record=%s", section_name, record_id
        )

    def apply_skill(
        self, record_id: str, skill_name: str, skill_content: str, skills_dir: Path
    ) -> None:
        """Apply a skill mutation by writing a skill file."""
        skills_dir.mkdir(parents=True, exist_ok=True)
        skill_path = skills_dir / f"{skill_name}.md"
        skill_path.write_text(skill_content, encoding="utf-8")
        self._registry.mark_applied(record_id, skill_content)
        logger.info("Applied skill evolution skill=%s path=%s", skill_name, skill_path)

    def rollback(self, record_id: str) -> bool:
        """Rollback a specific mutation.

        Returns:
            True if rollback succeeded.
        """
        record = self._registry.get_record(record_id)
        if record is None:
            logger.warning("Cannot rollback — record not found id=%s", record_id)
            return False

        if not record.applied_at:
            logger.warning("Cannot rollback — record was never applied id=%s", record_id)
            return False

        if (
            record.artifact_type == "tool_description"
            and record.artifact_id in self._description_overrides
        ):
            del self._description_overrides[record.artifact_id]
        elif (
            record.artifact_type == "prompt_section"
            and record.artifact_id in self._prompt_overrides
        ):
            del self._prompt_overrides[record.artifact_id]

        self._registry.mark_reverted(record_id)
        logger.info("Rolled back evolution record=%s artifact=%s", record_id, record.artifact_id)
        return True

    def get_tool_description(self, tool_name: str, default: str) -> str:
        """Get the effective tool description (override or default)."""
        return self._description_overrides.get(tool_name, default)

    def get_prompt_section(self, section_name: str, default: str) -> str:
        """Get the effective prompt section (override or default)."""
        return self._prompt_overrides.get(section_name, default)
