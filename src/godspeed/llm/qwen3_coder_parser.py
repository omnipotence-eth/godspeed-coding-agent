"""Post-processor for Qwen3-Coder tool-call XML.

Ollama's built-in tool-call parser (as of 0.20.x) does not recognize the
``<function=name>\\n<parameter=key>\\nvalue\\n</parameter>\\n</function>``
XML that Qwen3-Coder-30B-A3B-Instruct emits. The layers are all downloaded
and the model follows the advertised ``tools`` capability, but the
response comes back with the tool-call embedded in the ``content`` field
instead of in the structured ``tool_calls`` field.

This module extracts those blocks from response text and synthesizes
OpenAI-style tool_calls dicts so the rest of Godspeed's pipeline sees a
standard response.

Usage (from llm.client._call):

    parsed = extract_qwen3_coder_tool_calls(content_text)
    if parsed and not tool_calls:
        tool_calls = parsed
        content_text = ""  # tool-call text is not a real assistant reply

Scope:
- Handles multiple ``<function=...>`` blocks per response.
- Handles missing / extra whitespace inside the XML.
- Best-effort JSON coercion for each parameter value — strings,
  booleans, numbers, and bare JSON objects/arrays are recognized.
  Anything else is passed through as a string.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any

logger = logging.getLogger(__name__)

# Opening tag, then one or more <parameter=...> blocks, then closing tag.
# Greedy on body but non-greedy on parameters so each block matches once.
_FUNCTION_RE = re.compile(
    r"<function=([^>\s]+)>\s*(.*?)\s*</function>",
    re.DOTALL,
)
_PARAMETER_RE = re.compile(
    r"<parameter=([^>\s]+)>\s*(.*?)\s*</parameter>",
    re.DOTALL,
)


def looks_like_qwen3_coder_output(content: str) -> bool:
    """Cheap detector used to decide whether to even attempt parsing."""
    if not content:
        return False
    return "<function=" in content


def extract_qwen3_coder_tool_calls(content: str) -> list[dict[str, Any]]:
    """Parse every ``<function=...>`` block into an OpenAI-style tool_calls dict.

    Returns an empty list when no blocks are found; callers should treat
    that as "leave the response untouched."
    """
    if not looks_like_qwen3_coder_output(content):
        return []

    tool_calls: list[dict[str, Any]] = []
    for func_match in _FUNCTION_RE.finditer(content):
        name = func_match.group(1).strip()
        body = func_match.group(2)
        if not name:
            continue

        arguments: dict[str, Any] = {}
        for param_match in _PARAMETER_RE.finditer(body):
            key = param_match.group(1).strip()
            raw = param_match.group(2)
            if not key:
                continue
            arguments[key] = _coerce_parameter_value(raw)

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
        logger.info("Parsed %d Qwen3-Coder tool call(s) from response text", len(tool_calls))
    return tool_calls


def _coerce_parameter_value(raw: str) -> Any:
    """Best-effort JSON coercion for parameter values.

    Qwen3-Coder emits parameter values as raw text. Callers expect types
    matching the tool schema (strings, bools, numbers, nested objects).
    Try JSON first for anything that looks like a structured value,
    then fall back to the trimmed string.
    """
    value = raw.strip()
    if not value:
        return ""

    # Booleans + null first (case-insensitive).
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None

    # JSON-looking structured values.
    if value[0] in "[{" or (value[0] == '"' and value[-1] == '"'):
        try:
            return json.loads(value)
        except (ValueError, json.JSONDecodeError):
            pass

    # Numbers (int / float).
    if value[0].isdigit() or (value[0] == "-" and len(value) > 1 and value[1].isdigit()):
        try:
            if "." in value or "e" in value or "E" in value:
                return float(value)
            return int(value)
        except ValueError:
            pass

    # Anything else → raw string.
    return value
