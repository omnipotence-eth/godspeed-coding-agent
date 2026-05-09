"""Wiki → Skill bridge — auto-generate SKILL.md files from llm-wiki pages.

Scans ``llm-wiki/wiki/`` for tagged pages matching a topic filter, then
generates a godspeed-compatible SKILL.md with frontmatter and instructions.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


class WikiBridge:
    """Bridge between llm-wiki knowledge base and godspeed skills."""

    def __init__(self, wiki_dir: str | Path | None = None):
        self._wiki_dir = Path(wiki_dir) if wiki_dir else self._default_wiki()
        self._output_dir = Path.home() / ".godspeed" / "skills"

    @staticmethod
    def _default_wiki() -> Path:
        candidates = [
            Path.home() / "Documents" / "Project Portfolio" / "llm-wiki" / "wiki",
            Path.home() / "llm-wiki" / "wiki",
        ]
        for c in candidates:
            if c.is_dir():
                return c
        return candidates[0]

    def generate_skill(self, topic: str, output_name: str | None = None) -> Path | None:
        """Generate a skill from a wiki topic.

        Scans ``wiki/*.md`` for pages whose filename or frontmatter tags
        match ``topic``. Generates a SKILL.md.

        Returns the path to the generated skill directory, or None.
        """
        page = self._find_page(topic)
        if page is None:
            logger.warning("No wiki page found for topic=%s", topic)
            return None

        name = output_name or self._slugify(topic)
        content = page.read_text(encoding="utf-8", errors="replace")

        skill_dir = self._output_dir / name
        skill_dir.mkdir(parents=True, exist_ok=True)

        frontmatter = self._extract_frontmatter(content)
        description = frontmatter.get("short", frontmatter.get("title", f"Knowledge about {topic}"))
        body = self._strip_frontmatter(content)
        body = self._clean_body(body)

        skill_md = f"""---
name: {name}
description: {description}
trigger: {name}
version: "1.0.0"
metadata:
  source: llm-wiki
  topic: {topic}
  confidence: {frontmatter.get("confidence", "medium")}
---

{body}
"""
        skill_path = skill_dir / "SKILL.md"
        skill_path.write_text(skill_md)

        ref_dir = skill_dir / "references"
        ref_dir.mkdir(exist_ok=True)
        (ref_dir / "source.md").write_text(content)

        logger.info("Generated skill name=%s topic=%s from=%s", name, topic, page)
        return skill_dir

    def generate_all(self, tag_filter: str | None = None) -> list[Path]:
        """Generate skills for all wiki pages, optionally filtering by tag.

        Returns list of generated skill directory paths.
        """
        if not self._wiki_dir.is_dir():
            logger.warning("Wiki dir %s not found", self._wiki_dir)
            return []

        generated: list[Path] = []
        for page in sorted(self._wiki_dir.glob("*.md")):
            if tag_filter:
                raw = page.read_text(encoding="utf-8", errors="replace")
                frontmatter = self._extract_frontmatter(raw)
                tags = frontmatter.get("tags", [])
                tags_str = " ".join(tags).lower() if isinstance(tags, list) else str(tags).lower()
                if tag_filter.lower() not in tags_str:
                    continue

            topic = page.stem
            result = self.generate_skill(topic.replace("-", " ").title(), output_name=topic)
            if result:
                generated.append(result)

        logger.info("Generated %d skills from wiki (tag_filter=%s)", len(generated), tag_filter)
        return generated

    def _find_page(self, topic: str) -> Path | None:
        slug = self._slugify(topic)
        for page in self._wiki_dir.glob("*.md"):
            if page.stem == slug:
                return page
        for page in self._wiki_dir.glob("*.md"):
            if slug in page.stem:
                return page
        return None

    @staticmethod
    def _slugify(text: str) -> str:
        slug = text.lower().strip()
        slug = re.sub(r"[^a-z0-9\s_-]", "", slug)
        slug = re.sub(r"[\s_]+", "-", slug)
        slug = re.sub(r"-+", "-", slug)
        return slug.strip("-")

    @staticmethod
    def _extract_frontmatter(text: str) -> dict:
        stripped = text.strip()
        if not stripped.startswith("---"):
            return {}
        end = stripped.find("---", 3)
        if end == -1:
            return {}
        import yaml
        try:
            fm = yaml.safe_load(stripped[3:end].strip())
            return fm if isinstance(fm, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _strip_frontmatter(text: str) -> str:
        stripped = text.strip()
        if not stripped.startswith("---"):
            return text
        end = stripped.find("---", 3)
        if end == -1:
            return text
        return stripped[end + 3:].strip()

    @staticmethod
    def _clean_body(body: str) -> str:
        lines = body.split("\n")
        cleaned = []
        for line in lines:
            if line.startswith("---"):
                continue
            cleaned.append(line)
        return "\n".join(cleaned).strip()
