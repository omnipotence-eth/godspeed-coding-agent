"""Post-processor for JSON-in-markdown tool-call output.

Many local models (Qwen2.5-Coder via Ollama, DeepSeek, etc.) do not emit
native ``tool_calls`` arrays. Instead they wrap tool calls in markdown
JSON code blocks::

    ```json
    {"name": "file_write", "arguments": {"file_path": "...", "content": "..."}}
    ```

This module extracts those blocks from response text and synthesizes
OpenAI-style ``tool_calls`` dicts so the rest of Godspeed's pipeline
sees a standard response.

Usage (from llm.client._call):

    parsed = extract_json_markdown_tool_calls(content_text)
    if parsed and not tool_calls:
        tool_calls = parsed
        content_text = ""  # tool-call text is not a real assistant reply

Scope:
- Handles multiple JSON blocks per response.
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

# Fingerprint: look for name/arguments keys which indicate a tool call
_TOOL_CALL_FINGERPRINT_RE = re.compile(
    r'"(?:name|tool_name)"\s*:\s*"',
    re.IGNORECASE,
)


def looks_like_json_tool_call_output(content: str) -> bool:
    """Cheap detector: does the content contain JSON blocks with tool calls?"""
    if not content:
        return False
    if "```" not in content:
        return False
    return bool(_TOOL_CALL_FINGERPRINT_RE.search(content))


def extract_json_markdown_tool_calls(content: str) -> list[dict[str, Any]]:
    """Parse every JSON markdown block into OpenAI-style tool_calls dicts.

    Returns an empty list when no valid tool-call blocks are found.
    """
    if not looks_like_json_tool_call_output(content):
        return []

    tool_calls: list[dict[str, Any]] = []
    for block_match in _JSON_BLOCK_RE.finditer(content):
        raw_json = block_match.group(1).strip()
        if not raw_json:
            continue

        try:
            data = json.loads(raw_json)
        except (ValueError, json.JSONDecodeError):
            continue

        # Support both single object and list of objects
        items = data if isinstance(data, list) else [data]

        for item in items:
            if not isinstance(item, dict):
                continue
            name = item.get("name") or item.get("tool_name") or item.get("function")
            if not name or not isinstance(name, str):
                continue

            arguments = item.get("arguments") or item.get("args") or item.get("parameters") or {}
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except (ValueError, json.JSONDecodeError):
                    arguments = {"raw": arguments}
            if not isinstance(arguments, dict):
                arguments = {"raw": arguments}

            tool_calls.append(
                {
                    "id": f"call_{uuid.uuid4().hex[:10]}",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments),
                    },
                }
            )

    if tool_calls:
        logger.info(
            "Parsed %d JSON-markdown tool call(s) from response text",
            len(tool_calls),
        )
    return tool_calls
