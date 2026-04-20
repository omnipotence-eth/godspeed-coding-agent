"""Stage D — serialize one executed session to the final training record.

Thin adapter: ``ConversationLogger`` JSONL → ``TrainingExporter.export_session
(fmt="openai")`` → ``{messages, tools}`` dict → JSONL line on the output stream.

We always emit the full 21-tool schema list in ``tools`` so every training
sample sees the same toolbox the production agent sees at inference.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from godspeed.tools.registry import ToolRegistry
from godspeed.training.exporter import TrainingExporter

logger = logging.getLogger(__name__)


def emit_session(
    session_path: Path,
    registry: ToolRegistry,
    *,
    max_tool_output: int = 2000,
) -> dict[str, Any] | None:
    """Convert one session JSONL → one ``{messages, tools}`` training record.

    Returns ``None`` if the session is empty or unreadable.
    """
    exporter = TrainingExporter()
    record = exporter.export_session(
        session_path=session_path,
        fmt="openai",
        tool_schemas=registry.get_schemas(),
        max_tool_output=max_tool_output,
    )
    if record is None:
        return None

    # Sanity-check the shape matches what training's messages_raw reader
    # expects. Cheap; catches regressions early.
    if "messages" not in record:
        msg = f"exporter returned record without 'messages' key: {list(record)}"
        raise RuntimeError(msg)
    if "tools" not in record:
        msg = f"exporter returned record without 'tools' key (wrong fmt?): {list(record)}"
        raise RuntimeError(msg)
    if len(record["tools"]) != 21:
        logger.warning(
            "expected 21-tool schema in record but got %d — registry drift?",
            len(record["tools"]),
        )
    return record


def append_jsonl(record: dict[str, Any], output_path: Path) -> None:
    """Append one training record to an output JSONL stream."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def emit_and_append(
    session_path: Path,
    output_path: Path,
    registry: ToolRegistry,
    *,
    max_tool_output: int = 2000,
) -> bool:
    """One-shot: export + append. Returns True on success, False on empty/skip."""
    record = emit_session(session_path, registry, max_tool_output=max_tool_output)
    if record is None:
        return False
    append_jsonl(record, output_path)
    return True


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 3:
        sys.stderr.write(
            "usage: python -m experiments.phase_a1.emit <session.jsonl> <output.jsonl>\n"
        )
        sys.exit(2)

    from experiments.phase_a1.registry_builder import build_registry

    reg = build_registry()
    ok = emit_and_append(Path(sys.argv[1]), Path(sys.argv[2]), reg)
    logger.info("emitted=%s path=%s", ok, sys.argv[2])
