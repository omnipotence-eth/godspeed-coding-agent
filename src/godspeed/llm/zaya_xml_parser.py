"""Post-processor for ZAYA1-8B tool-call XML.

ZAYA1-8B outputs tool calls in JSON-inside-XML format via the vLLM
``--tool-call-parser zaya_xml`` parser. The model emits:

    <zyphra_tool_call>{"name": "tool", "arguments": {...}}</zyphra_tool_call>
    <zyphra_tool_response>result text</zyphra_tool_response>

Unlike Qwen3-Coder which uses ``<function=name>`` tags, ZAYA uses the
dedicated special tokens (101-104) for tool-call boundaries.

This module extracts those blocks from response text and synthesizes
OpenAI-style tool_calls dicts so the rest of Godspeed's pipeline sees a
standard response.

Usage (from llm.client._call):

    parsed = extract_zaya_tool_calls(content_text)
    if parsed and not tool_calls:
        tool_calls = parsed
        content_text = ""

Scope:
- Handles multiple ``<zyphra_tool_call>`` blocks per response.
- Handles missing / extra whitespace inside the XML.
- Validates JSON payloads; passes malformed ones as raw text.
- Supports ``<think>`` blocks — strips them from tool-call extraction.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any

logger = logging.getLogger(__name__)

_ZAYA_TOOL_CALL_RE = re.compile(
    r"<zyphra_tool_call>(.*?)</zyphra_tool_call>",
    re.DOTALL,
)

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def looks_like_zaya_output(content: str) -> bool:
    """Cheap detector for ZAYA XML tool-call output."""
    if not content:
        return False
    return "<zyphra_tool_call>" in content


def extract_zaya_tool_calls(content: str) -> list[dict[str, Any]]:
    """Parse every ``<zyphra_tool_call>`` block into OpenAI-style tool_calls.

    Args:
        content: Raw model output text, potentially containing multiple
                 tool-call blocks.

    Returns:
        List of OpenAI-compatible tool_calls dicts with ``id``, ``type``,
        ``function.name``, and ``function.arguments`` keys.
    """
    if not content:
        return []

    tool_calls: list[dict[str, Any]] = []

    for match in _ZAYA_TOOL_CALL_RE.finditer(content):
        payload = match.group(1).strip()
        if not payload:
            continue

        try:
            tc = json.loads(payload)
            name = tc.get("name", "")
            arguments = tc.get("arguments", {})

            if isinstance(arguments, dict):
                args_str = json.dumps(arguments, ensure_ascii=False)
            elif isinstance(arguments, str):
                try:
                    json.loads(arguments)
                    args_str = arguments
                except json.JSONDecodeError:
                    args_str = json.dumps({"_raw": arguments}, ensure_ascii=False)
            else:
                args_str = json.dumps(arguments, ensure_ascii=False)

            tool_calls.append(
                {
                    "id": f"call_{uuid.uuid4().hex[:24]}",
                    "type": "function",
                    "function": {"name": name, "arguments": args_str},
                }
            )
        except json.JSONDecodeError:
            logger.debug("Malformed ZAYA tool-call JSON: %s", payload[:200])

    return tool_calls


def strip_tool_calls_from_content(content: str) -> str:
    """Remove ``<zyphra_tool_call>`` blocks from content, keeping only
    natural-language text and ``<think>`` blocks.

    Returns:
        Content string with XML tool-call blocks removed and whitespace
        normalized.
    """
    if not content:
        return ""

    stripped = _ZAYA_TOOL_CALL_RE.sub("", content)
    stripped = re.sub(r"\n{3,}", "\n\n", stripped).strip()
    return stripped


def extract_thinking(content: str) -> str:
    """Extract thinking content from ``<think>`` blocks for logging.

    Returns:
        Concatenated thinking content across all think blocks, or empty
        string if no think blocks found.
    """
    if not content:
        return ""

    parts: list[str] = []
    for match in _THINK_BLOCK_RE.finditer(content):
        thinking = match.group(0)
        inner = thinking[len("<think>"): -len("</think>")]
        parts.append(inner.strip())

    return "\n".join(parts)
