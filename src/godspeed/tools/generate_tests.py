"""Generate-tests tool — LLM-driven pytest skeleton for a source file.

Reads a source file, asks the model to produce a pytest-compatible test
module, and writes it under ``tests/`` next to the source. Closes the
test-first discipline loop with one tool call.

Uses ``ToolContext.llm_client`` (added in v2.8.0) rather than requiring
a direct ``LLMClient`` reference — the tool stays decoupled from the
``llm/`` package.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from godspeed.tools.base import RiskLevel, Tool, ToolContext, ToolResult
from godspeed.tools.path_utils import resolve_tool_path

logger = logging.getLogger(__name__)

MAX_SOURCE_CHARS = 8000

GENERATE_TESTS_SYSTEM_PROMPT = """\
You are a senior engineer writing pytest tests. Given a source file, \
produce a complete, runnable pytest module that tests its public \
functions and classes.

Rules:
- Import only what you need; prefer `from module import thing`.
- One test function per behavior (happy path + one failure mode minimum).
- Use `pytest.raises` for expected exceptions.
- Use `tmp_path` fixture for filesystem tests.
- Use type hints on test functions.
- No mocking unless the code under test depends on I/O, network, or LLMs.
- No placeholder `pass` tests. Every test asserts something meaningful.
- No explanatory prose — return ONLY the test module source, ready to \
  save to disk.
"""


class GenerateTestsTool(Tool):
    """Generate a pytest test module for a source file.

    Reads ``source_path`` from the project, asks the LLM to write a
    pytest module, and writes the result to ``output_path`` (default:
    ``tests/test_<basename>.py`` next to the project root). Returns the
    path to the written file.

    Requires the ``ToolContext`` to have ``llm_client`` populated — the
    headless runner and TUI both wire it automatically.
    """

    @property
    def name(self) -> str:
        return "generate_tests"

    @property
    def description(self) -> str:
        return (
            "Generate a pytest test module for a source file using an LLM. "
            "Reads the source, writes tests to tests/test_<basename>.py. "
            "Returns the path to the generated test file — the agent should "
            "then run test_runner to confirm the generated tests actually "
            "pass against the current implementation.\n\n"
            "Example: generate_tests(source_path='src/myapp/util.py')\n"
            "Example: generate_tests(source_path='src/myapp/core.py', "
            "output_path='tests/test_core.py')"
        )

    @property
    def risk_level(self) -> RiskLevel:
        # Writes a file to disk; treat as LOW-risk write (user sees it in the
        # permission prompt before the tool executes in interactive mode).
        return RiskLevel.LOW

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "source_path": {
                    "type": "string",
                    "description": (
                        "Path to the source file to generate tests for (relative to project root)."
                    ),
                },
                "output_path": {
                    "type": "string",
                    "description": (
                        "Optional path for the generated test file. "
                        "Defaults to tests/test_<basename>.py relative to "
                        "the project root."
                    ),
                },
            },
            "required": ["source_path"],
        }

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        if context.llm_client is None:
            return ToolResult.failure(
                "generate_tests requires ToolContext.llm_client but it is "
                "unset. Use this tool from a headless or TUI session (not "
                "from a bare ToolContext fixture)."
            )

        source_arg = arguments.get("source_path", "")
        if not isinstance(source_arg, str) or not source_arg.strip():
            return ToolResult.failure("source_path must be a non-empty string")

        try:
            source_path = resolve_tool_path(source_arg, context.cwd)
        except ValueError as exc:
            return ToolResult.failure(str(exc))

        if not source_path.is_file():
            return ToolResult.failure(f"source_path does not exist: {source_arg}")

        try:
            source_text = source_path.read_text(encoding="utf-8")
        except OSError as exc:
            return ToolResult.failure(f"Could not read source file: {exc}")

        if len(source_text) > MAX_SOURCE_CHARS:
            return ToolResult.failure(
                f"Source file is {len(source_text)} chars; max is "
                f"{MAX_SOURCE_CHARS}. Split the file or target a smaller unit."
            )

        output_arg = arguments.get("output_path")
        output_path = _resolve_output_path(output_arg, source_path, context.cwd)

        module_name = _module_name_from(source_path, context.cwd)
        user_prompt = _build_user_prompt(module_name, source_text)

        try:
            response = await context.llm_client.chat(
                messages=[
                    {"role": "system", "content": GENERATE_TESTS_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ]
            )
        except Exception as exc:
            logger.warning("LLM call failed during generate_tests: %s", exc)
            return ToolResult.failure(f"LLM call failed: {exc}")

        test_source = _clean_llm_output(getattr(response, "content", "") or "")
        if not test_source.strip():
            return ToolResult.failure("LLM returned empty output")

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(test_source, encoding="utf-8")
        except OSError as exc:
            return ToolResult.failure(f"Could not write test file: {exc}")

        rel = _display_rel(output_path, context.cwd)
        line_count = test_source.count("\n") + 1
        logger.info("generate_tests wrote file=%s lines=%d", rel, line_count)
        return ToolResult.success(
            f"Generated {line_count} lines of tests at {rel}. "
            "Run test_runner to verify the tests pass."
        )


def _resolve_output_path(
    output_arg: Any,
    source_path: Path,
    cwd: Path,
) -> Path:
    """Resolve the output path, defaulting to tests/test_<basename>.py."""
    if isinstance(output_arg, str) and output_arg.strip():
        candidate = Path(output_arg)
        if candidate.is_absolute():
            return candidate
        return cwd / candidate
    stem = source_path.stem
    return cwd / "tests" / f"test_{stem}.py"


def _module_name_from(source_path: Path, cwd: Path) -> str:
    """Best-effort module path reconstruction for the system prompt.

    Falls back to the filename stem when the source isn't under cwd.
    """
    try:
        rel = source_path.relative_to(cwd)
    except ValueError:
        return source_path.stem
    parts = rel.with_suffix("").parts
    # Drop a leading `src` when the project uses the src-layout convention.
    if parts and parts[0] == "src":
        parts = parts[1:]
    return ".".join(parts) if parts else source_path.stem


def _build_user_prompt(module_name: str, source_text: str) -> str:
    return (
        f"Write a pytest module for `{module_name}`. Source file:\n\n"
        f"```python\n{source_text}\n```\n\n"
        "Return only the test module source."
    )


def _clean_llm_output(raw: str) -> str:
    """Strip leading/trailing markdown code fences if the LLM wrapped the output."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # Drop the opening fence line (``` or ```python)
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text + ("\n" if text and not text.endswith("\n") else "")


def _display_rel(path: Path, cwd: Path) -> str:
    try:
        return str(path.relative_to(cwd))
    except ValueError:
        return str(path)
