"""Ollama model management — pull, list, show, and delete models.

Provides both a Godspeed Tool (usable by the agent in TUI mode) and
a standalone module with functions callable from the CLI and slash commands.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any

from godspeed.tools.base import RiskLevel, Tool, ToolContext, ToolResult

logger = logging.getLogger(__name__)

OLLAMA_BIN = "ollama"
OLLAMA_API = "http://localhost:11434/api"


@dataclass
class OllamaModelInfo:
    """Information about a locally installed Ollama model."""

    name: str
    size_bytes: int = 0
    modified_at: str = ""
    digest: str = ""

    @property
    def size_gb(self) -> float:
        return self.size_bytes / (1024**3)


def _run_ollama(args: list[str], timeout: int = 300) -> tuple[int, str, str]:
    """Run an ollama CLI command. Returns (returncode, stdout, stderr)."""
    try:
        result = subprocess.run(
            [OLLAMA_BIN, *args],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode, result.stdout, result.stderr
    except FileNotFoundError:
        return -1, "", "ollama is not installed"
    except subprocess.TimeoutExpired:
        return -2, "", f"ollama {args[0]} timed out after {timeout}s"


def _ollama_installed() -> bool:
    return shutil.which(OLLAMA_BIN) is not None


def list_models() -> list[OllamaModelInfo]:
    """List all locally installed Ollama models."""
    if not _ollama_installed():
        return []

    rc, stdout, stderr = _run_ollama(["list"], timeout=10)
    if rc != 0:
        logger.warning("ollama list failed: %s", stderr)
        return []

    models: list[OllamaModelInfo] = []
    for line in stdout.strip().splitlines():
        parts = line.split()
        if not parts:
            continue
        name = parts[0]
        info = OllamaModelInfo(name=name)

        # Parse size and modified time from remaining columns
        for i, part in enumerate(parts[1:], 1):
            if part.endswith("GB") or part.endswith("MB"):
                try:
                    if part.endswith("GB"):
                        info.size_bytes = int(float(part[:-2]) * 1024**3)
                    elif part.endswith("MB"):
                        info.size_bytes = int(float(part[:-2]) * 1024**2)
                except ValueError:
                    pass
            elif "ago" in part and i + 1 < len(parts):
                info.modified_at = part + " " + parts[i + 1]
                break

        models.append(info)
    return models


def pull_model(model: str, on_progress: Any = None) -> bool:
    """Pull a model from Ollama. Returns True on success.

    Args:
        model: Model name to pull (e.g. 'rnj-1:8b').
        on_progress: Optional callback(status: str) for progress updates.
    """
    if not _ollama_installed():
        if on_progress:
            on_progress("Ollama is not installed")
        return False

    try:
        result = subprocess.run(
            [OLLAMA_BIN, "pull", model],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode == 0:
            return True
        logger.warning("ollama pull %s failed: %s", model, result.stderr)
        if on_progress:
            on_progress(f"Pull failed: {result.stderr[:200]}")
        return False
    except FileNotFoundError:
        if on_progress:
            on_progress("Ollama is not installed")
        return False
    except subprocess.TimeoutExpired:
        if on_progress:
            on_progress("Pull timed out after 10min")
        return False


async def pull_model_async(
    model: str,
    on_progress: Any = None,
) -> bool:
    """Pull a model from Ollama asynchronously with streaming progress.

    Uses ollama's JSON streaming API for real-time progress bars.
    """
    if not _ollama_installed():
        if on_progress:
            on_progress("Ollama is not installed")
        return False

    try:
        proc = await asyncio.create_subprocess_exec(
            OLLAMA_BIN,
            "pull",
            model,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout = proc.stdout
        if stdout is None:  # guaranteed by PIPE
            return False
        while True:
            line = await stdout.readline()
            if not line:
                break
            try:
                data = json.loads(line.decode().strip())
                if on_progress and data:
                    status = data.get("status", "")
                    if status:
                        if "completed" in data and "total" in data:
                            pct = (
                                int(data["completed"] / data["total"] * 100) if data["total"] else 0
                            )
                            tag = data.get("digest", "")[:12]
                            on_progress(f"Downloading {tag} {status} ({pct}%)")
                        else:
                            on_progress(status)
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass

        await proc.wait()
        return proc.returncode == 0

    except FileNotFoundError:
        if on_progress:
            on_progress("Ollama is not installed")
        return False


def delete_model(model: str) -> tuple[bool, str]:
    """Delete a model from Ollama. Returns (success, message)."""
    if not _ollama_installed():
        return False, "Ollama is not installed"

    rc, stdout, stderr = _run_ollama(["rm", model], timeout=30)
    if rc == 0:
        return True, stdout.strip() or f"Deleted {model}"
    return False, stderr.strip() or stdout.strip()


def show_model(model: str) -> dict[str, Any] | None:
    """Show detailed info about a model. Returns None if not found."""
    if not _ollama_installed():
        return None

    rc, stdout, _stderr = _run_ollama(["show", model], timeout=10)
    if rc != 0:
        return None

    # Parse the show output into a dict
    info: dict[str, Any] = {"name": model}
    for line in stdout.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            key, _, value = line.partition(" ")
            # Skip indented continuation lines (they're part of previous value)
            if line.startswith(" ") or line.startswith("\t"):
                continue
            info[key.lower()] = value.strip()
        except ValueError:
            continue
    return info


def is_model_installed(model: str) -> bool:
    """Check if a specific model is installed locally."""
    return any(m.name == model for m in list_models())


def ensure_model_pulled(model: str, on_progress: Any = None) -> bool:
    """Pull a model if it's not already installed. Returns True if available."""
    if is_model_installed(model):
        return True
    if on_progress:
        on_progress(f"Model {model} not found locally. Pulling from Ollama...")
    return pull_model(model, on_progress)


class OllamaTool(Tool):
    """Manage Ollama models — list, pull, show, and delete."""

    name: str = "ollama"
    description: str = (
        "Manage locally installed Ollama language models. "
        "Actions: list (show installed models), pull (download a model), "
        "show (model details), delete (remove a model). "
        "Use this to check what models are available before switching."
    )
    risk_level: RiskLevel = RiskLevel.LOW

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list", "pull", "show", "delete"],
                    "description": "Action to perform: list, pull, show, or delete.",
                },
                "model": {
                    "type": "string",
                    "description": (
                        "Model name for pull/show/delete actions. "
                        "Use the exact tag (e.g. 'rnj-1:8b', 'qwen2.5-coder:14b')."
                    ),
                },
            },
            "required": ["action"],
        }

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        action = arguments.get("action", "list")
        model = arguments.get("model", "")

        if action == "list":
            models = list_models()
            if not models:
                return ToolResult.ok("No local models found. Install Ollama and pull some models.")
            lines = [
                f"{m.name:40s} {m.size_gb:.1f} GB" for m in sorted(models, key=lambda m: m.name)
            ]
            return ToolResult.ok(f"Installed models ({len(models)}):\n" + "\n".join(lines))

        if action == "pull":
            if not model:
                return ToolResult.failure("model is required for pull action")
            loop = asyncio.get_running_loop()
            success = await loop.run_in_executor(None, pull_model, model)
            if success:
                return ToolResult.ok(f"Successfully pulled {model}")
            return ToolResult.failure(f"Failed to pull {model}")

        if action == "show":
            if not model:
                return ToolResult.failure("model is required for show action")
            info = show_model(model)
            if info is None:
                return ToolResult.failure(f"Model {model!r} not found")
            lines = [f"{k}: {v}" for k, v in info.items()]
            return ToolResult.ok(f"Model {model}:\n" + "\n".join(lines))

        if action == "delete":
            if not model:
                return ToolResult.failure("model is required for delete action")
            success, message = delete_model(model)
            if success:
                return ToolResult.ok(message)
            return ToolResult.failure(f"Failed to delete {model}: {message}")

        return ToolResult.failure(f"Unknown action: {action!r}")
