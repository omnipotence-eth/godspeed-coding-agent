"""Post-processor for JSON-in-markdown tool-call output.

Many local models (Qwen2.5-Coder via Ollama, DeepSeek, etc.) do not emit
native ``tool_calls`` arrays. Instead they wrap tool calls in markdown
JSON code blocks::

    ```json
    {"name": "file_write", "arguments": {"file_path": "...", "content": "..."}}
    ```

Some models (Qwen2.5-Coder via llama.cpp) emit bare JSON at the start of
the response without markdown fences::

    {"name": "file_write", "arguments": {"file_path": "...", "content": "..."}}

This module extracts both formats from response text and synthesizes
OpenAI-style ``tool_calls`` dicts so the rest of Godspeed's pipeline
sees a standard response.

Usage (from llm.client._call):

    parsed = extract_json_markdown_tool_calls(content_text)
    if parsed and not tool_calls:
        tool_calls = parsed
        content_text = ""  # tool-call text is not a real assistant reply

Scope:
- Handles multiple JSON blocks per response.
- Handles bare JSON objects at the beginning or end of content.
- Handles ``arguments`` as either a JSON object or a JSON string.
- Best-effort: returns empty list when no blocks are found.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any

logger = logging.getLogger(__name__)

# Match markdown JSON code blocks (```json ... ```)
_JSON_BLOCK_RE = re.compile(
    r"```(?:json)?\s*\n(.*?)\n\s*```",
    re.DOTALL,
)

# Match bare JSON tool-call objects — either at the start of content
# (after optional whitespace) or preceded by a newline/start boundary.
# This handles Qwen2.5-Coder via llama.cpp which emits:
#   {"name": "file_write", "arguments": {"file_path": "..."}}
# without markdown fences.
_BARE_JSON_TOOL_CALL_RE = re.compile(
    r'(?:^|\n)\s*\{\s*"(?:name|tool_name)"\s*:',
    re.MULTILINE,
)

# Fingerprint: look for name/arguments keys which indicate a tool call
_TOOL_CALL_FINGERPRINT_RE = re.compile(
    r'"(?:name|tool_name)"\s*:\s*"',
    re.IGNORECASE,
)


def looks_like_json_tool_call_output(content: str) -> bool:
    """Cheap detector: does the content contain JSON blocks with tool calls?"""
    if not content:
        return False
    return bool(_TOOL_CALL_FINGERPRINT_RE.search(content))


def _parse_tool_call_item(item: dict[str, Any]) -> dict[str, Any] | None:
    """Parse a single tool-call JSON object into the standard format.

    Returns None if the object doesn't look like a valid tool call.
    """
    name = item.get("name") or item.get("tool_name") or item.get("function")
    if not name or not isinstance(name, str):
        return None

    arguments = item.get("arguments") or item.get("args") or item.get("parameters") or {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except (ValueError, json.JSONDecodeError):
            arguments = {"raw": arguments}
    if not isinstance(arguments, dict):
        arguments = {"raw": arguments}

    return {
        "id": f"call_{uuid.uuid4().hex[:10]}",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments),
        },
    }


def _parse_json_text(raw_json: str) -> list[dict[str, Any]]:
    """Try to parse a JSON string into tool-call objects.

    Returns a list of parsed tool calls (may be empty).
    """
    try:
        data = json.loads(raw_json)
    except (ValueError, json.JSONDecodeError):
        return []

    # Support both single object and list of objects
    items = data if isinstance(data, list) else [data]
    result: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        parsed = _parse_tool_call_item(item)
        if parsed:
            result.append(parsed)
    return result


def _strip_fenced_blocks(content: str) -> str:
    """Replace markdown fenced code blocks with whitespace.

    This prevents bare-JSON extraction from double-counting content
    that was already extracted by markdown-block phase 1.
    """
    return _JSON_BLOCK_RE.sub(lambda m: " " * len(m.group(0)), content)


def _extract_bare_json_tool_calls(content: str) -> list[dict[str, Any]]:
    """Extract bare JSON tool calls that aren't inside markdown fences.

    Qwen2.5-Coder via llama.cpp emits tool calls as bare JSON at the
    start of the content, e.g.:
      {"name": "file_write", "arguments": {"file_path": "test.txt", ...}}
    without any markdown backtick fences.
    """
    tool_calls: list[dict[str, Any]] = []

    # Strip markdown blocks first so we don't double-count
    bare_content = _strip_fenced_blocks(content)
    if not bare_content.strip():
        return tool_calls

    # Strategy 1: Try parsing the entire content as a single tool-call JSON
    full_parsed = _parse_json_text(bare_content.strip())
    if full_parsed:
        return full_parsed

    # Strategy 2: Try to extract JSON objects using the bare JSON pattern
    for match in _BARE_JSON_TOOL_CALL_RE.finditer(bare_content):
        # Find the matching closing brace
        start = match.start()
        depth = 0
        pos = start
        in_str = False
        escape = False
        while pos < len(bare_content):
            ch = bare_content[pos]
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"' and not escape:
                in_str = not in_str
            elif not in_str:
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = bare_content[start : pos + 1]
                        parsed = _parse_json_text(candidate)
                        if parsed:
                            tool_calls.extend(parsed)
                        break
            pos += 1

    return tool_calls


def extract_json_markdown_tool_calls(content: str) -> list[dict[str, Any]]:
    """Parse every JSON tool-call block into OpenAI-style tool_calls dicts.

    Supports:
    - Markdown fenced blocks (```json ... ```)
    - Bare JSON objects (``{"name": ..., "arguments": ...}``)
    - Both single objects and lists of objects

    Returns an empty list when no valid tool-call blocks are found.
    """
    if not looks_like_json_tool_call_output(content):
        return []

    tool_calls: list[dict[str, Any]] = []

    # Phase 1: extract from markdown code blocks
    for block_match in _JSON_BLOCK_RE.finditer(content):
        raw_json = block_match.group(1).strip()
        if not raw_json:
            continue
        tool_calls.extend(_parse_json_text(raw_json))

    # Phase 2: extract bare JSON (not inside backticks)
    tool_calls.extend(_extract_bare_json_tool_calls(content))

    if tool_calls:
        logger.info(
            "Parsed %d JSON-markdown tool call(s) from response text",
            len(tool_calls),
        )
    return tool_calls
