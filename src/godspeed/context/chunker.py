"""File chunking for codebase indexing — AST-based for Python, sliding window for others."""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_MAX_TOKENS = 512
OVERLAP_TOKENS = 128


@dataclass(frozen=True, slots=True)
class Chunk:
    """A chunk of source code from a file."""

    content: str
    file_path: str
    start_line: int
    end_line: int


def chunk_file(path: Path, max_tokens: int = DEFAULT_MAX_TOKENS) -> list[Chunk]:
    """Split a file into chunks for indexing.

    Python files are split by top-level functions/classes via AST.
    Other files use a sliding window approach.

    Args:
        path: Path to the file.
        max_tokens: Approximate max tokens per chunk (word-based estimate).

    Returns:
        List of Chunk objects.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.warning("Cannot read file %s: %s", path, exc)
        return []

    if not text.strip():
        return []

    file_str = str(path)

    if path.suffix in (".py", ".pyi"):
        chunks = _chunk_python(text, file_str, max_tokens)
        if chunks:
            return chunks

    return _chunk_sliding_window(text, file_str, max_tokens)


def _chunk_python(text: str, file_path: str, max_tokens: int) -> list[Chunk]:
    """Split Python files by top-level definitions (functions, classes)."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []

    lines = text.splitlines(keepends=True)
    chunks: list[Chunk] = []

    # Collect top-level node ranges
    nodes: list[tuple[int, int]] = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = node.lineno - 1  # 0-based
            end = node.end_lineno or node.lineno
            nodes.append((start, end))

    if not nodes:
        # No top-level definitions — treat as single chunk
        return _chunk_sliding_window(text, file_path, max_tokens)

    # Add module-level code before first definition
    if nodes[0][0] > 0:
        header = "".join(lines[: nodes[0][0]])
        if header.strip():
            chunks.append(
                Chunk(
                    content=header,
                    file_path=file_path,
                    start_line=1,
                    end_line=nodes[0][0],
                )
            )

    # Each top-level definition
    for start, end in nodes:
        content = "".join(lines[start:end])
        if content.strip():
            # If chunk is too large, split with sliding window
            word_count = len(content.split())
            if word_count > max_tokens:
                sub_chunks = _chunk_sliding_window(
                    content, file_path, max_tokens, line_offset=start
                )
                chunks.extend(sub_chunks)
            else:
                chunks.append(
                    Chunk(
                        content=content,
                        file_path=file_path,
                        start_line=start + 1,
                        end_line=end,
                    )
                )

    return chunks


def _chunk_sliding_window(
    text: str,
    file_path: str,
    max_tokens: int,
    line_offset: int = 0,
) -> list[Chunk]:
    """Split text into overlapping chunks by line count.

    Uses word count as a token estimate. Overlap ensures context
    is preserved across chunk boundaries.
    """
    lines = text.splitlines(keepends=True)
    if not lines:
        return []

    chunks: list[Chunk] = []
    i = 0

    while i < len(lines):
        # Accumulate lines until we hit max_tokens
        chunk_lines: list[str] = []
        word_count = 0
        start_idx = i

        while i < len(lines) and word_count < max_tokens:
            chunk_lines.append(lines[i])
            word_count += len(lines[i].split())
            i += 1

        content = "".join(chunk_lines)
        if content.strip():
            chunks.append(
                Chunk(
                    content=content,
                    file_path=file_path,
                    start_line=start_idx + line_offset + 1,
                    end_line=i + line_offset,
                )
            )

        # Overlap: back up by OVERLAP_TOKENS worth of lines
        if i < len(lines):
            overlap_words = 0
            backup = 0
            for j in range(len(chunk_lines) - 1, -1, -1):
                overlap_words += len(chunk_lines[j].split())
                backup += 1
                if overlap_words >= OVERLAP_TOKENS:
                    break
            i = max(start_idx + 1, i - backup)

    return chunks
