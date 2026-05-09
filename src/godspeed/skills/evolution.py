"""Skill evolution — lesson tracking and automatic skill rewriting (AutoContext pattern).

Lessons are corrections and observations accumulated per skill. When enough
high-confidence lessons exist, ``/skill-evolve`` rewrites the skill's
``SKILL.md``, weaving validated lessons into the content.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Lesson:
    """A single correction or observation tied to a skill."""

    skill_name: str
    text: str
    confidence: float = 1.0
    created_at: str = ""
    source: str = ""  # "user-correction", "self-observation", "session-analysis"


class SkillEvolution:
    """Tracks lessons per skill and generates evolved skill content."""

    def __init__(self, base_dir: Path | None = None):
        self._base = base_dir or Path.home() / ".godspeed" / "evolution"
        self._lessons_dir = self._base / "lessons"
        self._backups_dir = self._base / "backups"
        self._lessons_dir.mkdir(parents=True, exist_ok=True)
        self._backups_dir.mkdir(parents=True, exist_ok=True)

    def record_lesson(self, lesson: Lesson) -> None:
        lessons = self._load_lessons(lesson.skill_name)
        lessons.append({
            "text": lesson.text,
            "confidence": lesson.confidence,
            "created_at": lesson.created_at or datetime.now(tz=UTC).isoformat(),
            "source": lesson.source,
        })
        self._save_lessons(lesson.skill_name, lessons)
        logger.info("Recorded lesson for skill=%s confidence=%.1f", lesson.skill_name, lesson.confidence)

    def get_eligible_lessons(self, skill_name: str, min_confidence: float = 1.5) -> list[dict]:
        """Return lessons with cumulative confidence >= min_confidence."""
        lessons = self._load_lessons(skill_name)
        merged = self._merge_duplicates(lessons)
        return [l for l in merged if l.get("confidence", 0) >= min_confidence]

    def evolve(self, skill_name: str, skill_path: Path) -> bool:
        """Rewrite SKILL.md with eligible lessons folded in.

        Creates a timestamped backup before modifying. Returns True if
        the skill was updated.
        """
        eligible = self.get_eligible_lessons(skill_name)
        if not eligible:
            logger.info("No eligible lessons for skill=%s", skill_name)
            return False

        try:
            original = skill_path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.error("Cannot read skill=%s: %s", skill_name, exc)
            return False

        backup = self._backups_dir / f"{skill_name}_{datetime.now(tz=UTC).strftime('%Y%m%d_%H%M%S')}.md"
        backup.write_text(original)

        evolved = self._apply_lessons(original, eligible)
        skill_path.write_text(evolved)

        self._mark_folded(skill_name, eligible)
        logger.info("Evolved skill=%s (%d lessons folded, backup=%s)", skill_name, len(eligible), backup.name)
        return True

    def _load_lessons(self, skill_name: str) -> list[dict]:
        path = self._lessons_dir / f"{skill_name}.json"
        try:
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return []

    def _save_lessons(self, skill_name: str, lessons: list[dict]) -> None:
        path = self._lessons_dir / f"{skill_name}.json"
        path.write_text(json.dumps(lessons, indent=2))

    def _merge_duplicates(self, lessons: list[dict]) -> list[dict]:
        by_text: dict[str, list[dict]] = defaultdict(list)
        for l in lessons:
            norm = l.get("text", "").strip().lower()
            if norm:
                by_text[norm].append(l)

        result = []
        for text, group in by_text.items():
            result.append({
                "text": group[0]["text"],
                "confidence": sum(g.get("confidence", 1.0) for g in group),
                "created_at": group[-1]["created_at"],
                "source": group[-1].get("source", ""),
            })
        return result

    def _apply_lessons(self, original: str, lessons: list[dict]) -> str:
        lines = original.split("\n")
        inserts = ["", "## Lessons", ""]
        for l in lessons:
            inserts.append(f"- {l['text']}")
        inserts.append("")

        if "## Lessons" in original:
            return original

        return original.rstrip("\n") + "\n" + "\n".join(inserts) + "\n"

    def _mark_folded(self, skill_name: str, folded: list[dict]) -> None:
        lessons = self._load_lessons(skill_name)
        kept = []
        for l in lessons:
            l_norm = l.get("text", "").strip().lower()
            matched = any(f.get("text", "").strip().lower() == l_norm for f in folded)
            if not matched:
                kept.append(l)
        if kept:
            self._save_lessons(skill_name, kept)
        else:
            self._lessons_dir.joinpath(f"{skill_name}.json").write_text("[]")
