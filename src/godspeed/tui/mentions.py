"""Parse and resolve @-mentions in user input.

Supports @file:path, @folder:path, and @web:url syntax.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from godspeed.tools.path_utils import resolve_tool_path

logger = logging.getLogger(__name__)

# Match @file:path, @folder:path, @web:url — captures type and target
_MENTION_PATTERN = re.compile(r"@(file|folder|web):(\S+)")


@dataclass(frozen=True)
class Mention:
    """A parsed @-mention from user input."""

    type: Literal["file", "folder", "web"]
    raw: str  # Original matched text (e.g. "@file:src/main.py")
    target: str  # The path or URL after the colon


def parse_mentions(text: str) -> tuple[str, list[Mention]]:
    """Extract @-mentions from user input text.

    Args:
        text: Raw user input string.

    Returns:
        Tuple of (cleaned_text, mentions) where cleaned_text has mentions
        stripped and mentions is a list of parsed Mention objects.
    """
    mentions: list[Mention] = []
    for match in _MENTION_PATTERN.finditer(text):
        mention_type = match.group(1)
        target = match.group(2)
        mentions.append(
            Mention(
                type=mention_type,  # type: ignore[arg-type]
                raw=match.group(0),
                target=target,
            )
        )

    cleaned = _MENTION_PATTERN.sub("", text).strip()
    # Collapse multiple spaces left by stripping
    cleaned = re.sub(r" {2,}", " ", cleaned)
    return cleaned, mentions


async def resolve_mentions(
    mentions: list[Mention],
    cwd: Path,
) -> list[dict[str, Any]]:
    """Resolve mentions to content blocks for multimodal messages.

    Args:
        mentions: List of parsed Mention objects.
        cwd: Project working directory for path resolution.

    Returns:
        List of content blocks (text blocks with resolved content).
    """
    blocks: list[dict[str, Any]] = []

    for mention in mentions:
        try:
            if mention.type == "file":
                content = _resolve_file(mention.target, cwd)
                blocks.append(
                    {
                        "type": "text",
                        "text": f"[Content of {mention.target}]:\n{content}",
                    }
                )
            elif mention.type == "folder":
                content = _resolve_folder(mention.target, cwd)
                blocks.append(
                    {
                        "type": "text",
                        "text": f"[Directory listing of {mention.target}]:\n{content}",
                    }
                )
            elif mention.type == "web":
                content = await _resolve_web(mention.target)
                blocks.append(
                    {
                        "type": "text",
                        "text": f"[Content of {mention.target}]:\n{content}",
                    }
                )
        except (ValueError, OSError) as exc:
            logger.warning("Failed to resolve mention %s: %s", mention.raw, exc)
            blocks.append(
                {
                    "type": "text",
                    "text": f"[Error resolving {mention.raw}]: {exc}",
                }
            )

    return blocks


def _resolve_file(target: str, cwd: Path) -> str:
    """Read file content with path traversal protection."""
    resolved = resolve_tool_path(target, cwd)
    if not resolved.is_file():
        msg = f"Not a file: {target}"
        raise ValueError(msg)
    return resolved.read_text(encoding="utf-8", errors="replace")


def _resolve_folder(target: str, cwd: Path) -> str:
    """List directory contents with path traversal protection."""
    resolved = resolve_tool_path(target, cwd)
    if not resolved.is_dir():
        msg = f"Not a directory: {target}"
        raise ValueError(msg)

    entries = sorted(resolved.iterdir())
    lines = []
    for entry in entries[:100]:  # Cap at 100 entries
        suffix = "/" if entry.is_dir() else ""
        rel = entry.relative_to(cwd.resolve()) if entry.is_relative_to(cwd.resolve()) else entry
        lines.append(f"{rel}{suffix}")
    if len(entries) > 100:
        lines.append(f"... and {len(entries) - 100} more entries")
    return "\n".join(lines)


async def _resolve_web(url: str) -> str:
    """Fetch web content. HTTPS only, with size limit."""
    if not url.startswith("https://"):
        msg = f"Only HTTPS URLs are allowed, got: {url}"
        raise ValueError(msg)

    try:
        import httpx
    except ImportError as exc:
        msg = "httpx is required for @web mentions (pip install httpx)"
        raise ImportError(msg) from exc

    async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
        response = await client.get(url, headers={"User-Agent": "Godspeed/2.0"})
        response.raise_for_status()

        # Size limit: 100KB
        content = response.text
        if len(content) > 100_000:
            content = content[:100_000] + "\n... [truncated at 100KB]"
        return content
