"""Training data exporter — convert conversation logs to fine-tuning JSONL.

Reads per-session ``*.conversation.jsonl`` files produced by
``ConversationLogger`` and transforms them into formats consumed by
fine-tuning frameworks:

* **openai** — OpenAI fine-tuning format with ``tools`` array.
* **chatml** — ChatML with ``<|im_start|>`` tokens for Qwen/Mistral.
* **sharegpt** — ShareGPT ``conversations`` format for Unsloth dataset loading.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ExportFilters:
    """Criteria for including/excluding sessions from export."""

    min_tool_calls: int = 1
    success_only: bool = False
    min_turns: int = 2
    tools: list[str] | None = None
    max_sessions: int = 0  # 0 = unlimited


@dataclass(slots=True)
class ExportStats:
    """Summary statistics from an export run."""

    sessions_scanned: int = 0
    sessions_exported: int = 0
    sessions_filtered: int = 0
    total_messages: int = 0
    total_tool_calls: int = 0
    errors: list[str] = field(default_factory=list)


def _load_session(session_path: Path) -> list[dict[str, Any]]:
    """Read a conversation JSONL file into a list of message dicts."""
    messages: list[dict[str, Any]] = []
    with open(session_path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                messages.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning(
                    "Skipping malformed line %d in %s: %s", line_no, session_path.name, exc
                )
    return messages


def _count_tool_calls(messages: list[dict[str, Any]]) -> int:
    """Count total tool_calls across all assistant messages."""
    total = 0
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            total += len(msg["tool_calls"])
    return total


def _count_user_turns(messages: list[dict[str, Any]]) -> int:
    """Count user messages (each represents a conversational turn)."""
    return sum(1 for msg in messages if msg.get("role") == "user")


def _has_errors(messages: list[dict[str, Any]]) -> bool:
    """Check if any tool result has is_error=True."""
    return any(msg.get("role") == "tool" and msg.get("is_error") for msg in messages)


def _tools_used(messages: list[dict[str, Any]]) -> set[str]:
    """Collect the set of tool names used in the session."""
    names: set[str] = set()
    for msg in messages:
        if msg.get("role") == "tool" and msg.get("name"):
            names.add(msg["name"])
    return names


def _passes_filters(messages: list[dict[str, Any]], filters: ExportFilters) -> bool:
    """Check if a session passes the export filters."""
    if _count_user_turns(messages) < filters.min_turns:
        return False
    if _count_tool_calls(messages) < filters.min_tool_calls:
        return False
    if filters.success_only and _has_errors(messages):
        return False
    if filters.tools:
        used = _tools_used(messages)
        if not used.intersection(filters.tools):
            return False
    return True


def _truncate_content(content: str, max_chars: int) -> str:
    """Truncate tool output content to max_chars."""
    if len(content) <= max_chars:
        return content
    return content[:max_chars] + f"\n... (truncated, {len(content)} total chars)"


# ---------------------------------------------------------------------------
# OpenAI format
# ---------------------------------------------------------------------------


def _to_openai(
    messages: list[dict[str, Any]],
    tool_schemas: list[dict[str, Any]] | None = None,
    max_tool_output: int = 2000,
) -> dict[str, Any]:
    """Convert raw conversation messages to OpenAI fine-tuning format.

    Returns a single training example dict with ``messages`` and ``tools`` keys.
    """
    out_messages: list[dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role", "")

        # Skip meta events (compaction, etc.)
        if role == "meta":
            continue

        if role == "system":
            out_messages.append({"role": "system", "content": msg.get("content", "")})

        elif role == "user":
            content = msg.get("content", "")
            if isinstance(content, list):
                # Multimodal — flatten to text for fine-tuning
                text_parts = [
                    block.get("text", "") for block in content if block.get("type") == "text"
                ]
                content = "\n".join(text_parts)
            out_messages.append({"role": "user", "content": content})

        elif role == "assistant":
            assistant_msg: dict[str, Any] = {"role": "assistant"}
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls")

            if tool_calls:
                assistant_msg["content"] = content or None
                openai_calls = []
                for tc in tool_calls:
                    args = tc.get("arguments", tc.get("function", {}).get("arguments", {}))
                    if isinstance(args, dict):
                        args = json.dumps(args, ensure_ascii=False)
                    openai_calls.append(
                        {
                            "id": tc.get("id", tc.get("tool_call_id", "")),
                            "type": "function",
                            "function": {
                                "name": tc.get("name", tc.get("function", {}).get("name", "")),
                                "arguments": args,
                            },
                        }
                    )
                assistant_msg["tool_calls"] = openai_calls
            else:
                assistant_msg["content"] = content

            out_messages.append(assistant_msg)

        elif role == "tool":
            tool_msg: dict[str, Any] = {
                "role": "tool",
                "tool_call_id": msg.get("tool_call_id", ""),
                "content": _truncate_content(msg.get("content", ""), max_tool_output),
            }
            out_messages.append(tool_msg)

    result: dict[str, Any] = {"messages": out_messages}
    if tool_schemas:
        result["tools"] = tool_schemas
    return result


# ---------------------------------------------------------------------------
# ChatML format
# ---------------------------------------------------------------------------


def _to_chatml(
    messages: list[dict[str, Any]],
    max_tool_output: int = 2000,
) -> dict[str, Any]:
    """Convert to ChatML format with ``<|im_start|>`` tokens.

    Used by Qwen/Mistral native templates.
    """
    parts: list[str] = []

    for msg in messages:
        role = msg.get("role", "")
        if role == "meta":
            continue

        if role == "system":
            parts.append(f"<|im_start|>system\n{msg.get('content', '')}<|im_end|>")

        elif role == "user":
            content = msg.get("content", "")
            if isinstance(content, list):
                text_parts = [
                    block.get("text", "") for block in content if block.get("type") == "text"
                ]
                content = "\n".join(text_parts)
            parts.append(f"<|im_start|>user\n{content}<|im_end|>")

        elif role == "assistant":
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                tc_block = json.dumps(tool_calls, ensure_ascii=False)
                body = (
                    f"{content}\n<tool_call>\n{tc_block}\n</tool_call>"
                    if content
                    else (f"<tool_call>\n{tc_block}\n</tool_call>")
                )
                parts.append(f"<|im_start|>assistant\n{body}<|im_end|>")
            else:
                parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")

        elif role == "tool":
            truncated = _truncate_content(msg.get("content", ""), max_tool_output)
            name = msg.get("name", "unknown")
            parts.append(
                f'<|im_start|>tool\n<tool_response name="{name}">\n'
                f"{truncated}\n</tool_response><|im_end|>"
            )

    return {"text": "\n".join(parts)}


# ---------------------------------------------------------------------------
# ShareGPT format
# ---------------------------------------------------------------------------


def _to_sharegpt(
    messages: list[dict[str, Any]],
    max_tool_output: int = 2000,
) -> dict[str, Any]:
    """Convert to ShareGPT format (Unsloth-compatible).

    Returns ``{"conversations": [{"from": ..., "value": ...}, ...]}``.
    """
    conversations: list[dict[str, str]] = []

    for msg in messages:
        role = msg.get("role", "")
        if role == "meta":
            continue

        if role == "system":
            conversations.append({"from": "system", "value": msg.get("content", "")})

        elif role == "user":
            content = msg.get("content", "")
            if isinstance(content, list):
                text_parts = [
                    block.get("text", "") for block in content if block.get("type") == "text"
                ]
                content = "\n".join(text_parts)
            conversations.append({"from": "human", "value": content})

        elif role == "assistant":
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                tc_text = json.dumps(tool_calls, ensure_ascii=False)
                value = (
                    f"{content}\n\n[Tool Calls]\n{tc_text}"
                    if content
                    else (f"[Tool Calls]\n{tc_text}")
                )
                conversations.append({"from": "gpt", "value": value})
            else:
                conversations.append({"from": "gpt", "value": content})

        elif role == "tool":
            truncated = _truncate_content(msg.get("content", ""), max_tool_output)
            name = msg.get("name", "unknown")
            conversations.append({"from": "tool", "value": f"[{name}]\n{truncated}"})

    return {"conversations": conversations}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_FORMAT_MAP = {
    "openai": _to_openai,
    "chatml": _to_chatml,
    "sharegpt": _to_sharegpt,
}

SUPPORTED_FORMATS = frozenset(_FORMAT_MAP.keys())


class TrainingExporter:
    """Convert conversation logs to fine-tuning JSONL."""

    def export_session(
        self,
        session_path: Path,
        fmt: str = "openai",
        tool_schemas: list[dict[str, Any]] | None = None,
        max_tool_output: int = 2000,
    ) -> dict[str, Any] | None:
        """Convert one session JSONL → one training example.

        Returns None if the session file is empty or unreadable.
        """
        if fmt not in SUPPORTED_FORMATS:
            msg = f"Unsupported format: {fmt}. Use one of: {', '.join(sorted(SUPPORTED_FORMATS))}"
            raise ValueError(msg)

        messages = _load_session(session_path)
        if not messages:
            return None

        if fmt == "openai":
            return _to_openai(messages, tool_schemas=tool_schemas, max_tool_output=max_tool_output)
        if fmt == "chatml":
            return _to_chatml(messages, max_tool_output=max_tool_output)
        return _to_sharegpt(messages, max_tool_output=max_tool_output)

    def export_all(
        self,
        training_dir: Path,
        output_path: Path,
        fmt: str = "openai",
        filters: ExportFilters | None = None,
        tool_schemas: list[dict[str, Any]] | None = None,
        max_tool_output: int = 2000,
    ) -> ExportStats:
        """Batch export all sessions with optional filtering.

        Writes one JSON object per line to ``output_path``.
        """
        if fmt not in SUPPORTED_FORMATS:
            msg = f"Unsupported format: {fmt}. Use one of: {', '.join(sorted(SUPPORTED_FORMATS))}"
            raise ValueError(msg)

        if filters is None:
            filters = ExportFilters()

        stats = ExportStats()
        session_files = sorted(training_dir.glob("*.conversation.jsonl"))

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as out:
            for session_file in session_files:
                stats.sessions_scanned += 1

                try:
                    messages = _load_session(session_file)
                except Exception as exc:
                    stats.errors.append(f"{session_file.name}: {exc}")
                    continue

                if not messages:
                    stats.sessions_filtered += 1
                    continue

                if not _passes_filters(messages, filters):
                    stats.sessions_filtered += 1
                    continue

                # Convert
                example = self.export_session(
                    session_file,
                    fmt=fmt,
                    tool_schemas=tool_schemas,
                    max_tool_output=max_tool_output,
                )
                if example is None:
                    stats.sessions_filtered += 1
                    continue

                line = json.dumps(example, ensure_ascii=False, separators=(",", ":"))
                out.write(line + "\n")

                stats.sessions_exported += 1
                stats.total_messages += len(messages)
                stats.total_tool_calls += _count_tool_calls(messages)

                if 0 < filters.max_sessions <= stats.sessions_exported:
                    break

        logger.info(
            "Export complete scanned=%d exported=%d filtered=%d format=%s",
            stats.sessions_scanned,
            stats.sessions_exported,
            stats.sessions_filtered,
            fmt,
        )
        return stats
