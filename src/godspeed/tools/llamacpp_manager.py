"""llama.cpp server management for Godspeed — auto-start, health checks, model discovery.

Mirrors the Ollama manager pattern but for a local llama.cpp OpenAI-compatible
server. The server is expected to expose http://localhost:8080/v1 by default.

Usage from Godspeed CLI:
    _ensure_llamacpp()  # auto-start if not running

Usage from agent (TUI):
    llamacpp status     # check if server is up
    llamacpp start      # start the server
    llamacpp stop       # stop the server
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from godspeed.tools.base import RiskLevel, Tool, ToolContext, ToolResult

logger = logging.getLogger(__name__)

DEFAULT_URL = "http://127.0.0.1:8080"
DEFAULT_API_BASE = f"{DEFAULT_URL}/v1"
LLAMACPP_API_BASE_ENV = "LLAMACPP_API_BASE"

# Default paths — searched in order
DEFAULT_MODELS_DIR = Path.home() / ".llamacpp" / "models"
DEFAULT_SERVER_BINS: list[Path] = [
    Path.home() / "llama.cpp" / "build" / "bin" / "Release" / "llama-server.exe",
    Path.home() / "llama.cpp" / "build" / "bin" / "llama-server",
    Path.home() / "llama.cpp" / "build" / "Release" / "llama-server.exe",
]

# Model configuration for auto-start
DEFAULT_MODEL_FILE = "Qwen3.6-27B-UD-Q4_K_XL.gguf"
DEFAULT_CONTEXT = 32768
DEFAULT_GPU_LAYERS = 999


def _find_server_binary() -> Path | None:
    """Locate llama-server binary in common paths."""
    for path in DEFAULT_SERVER_BINS:
        if path.exists():
            return path
    # Try PATH
    which = shutil.which("llama-server")
    if which:
        return Path(which)
    return None


def _find_model() -> Path | None:
    """Locate the default GGUF model file."""
    candidate = DEFAULT_MODELS_DIR / DEFAULT_MODEL_FILE
    if candidate.exists():
        return candidate
    # Search models dir for any .gguf
    if DEFAULT_MODELS_DIR.exists():
        ggufs = sorted(DEFAULT_MODELS_DIR.glob("*.gguf"))
        if ggufs:
            return ggufs[0]
    return None


def is_server_running(url: str = DEFAULT_URL) -> bool:
    """Check if the llama.cpp server is reachable."""
    try:
        import urllib.request

        req = urllib.request.Request(f"{url}/v1/models", method="GET")  # noqa: S310
        with urllib.request.urlopen(req, timeout=2):  # noqa: S310
            return True
    except Exception:
        return False


def start_server(
    *,
    model_path: Path | None = None,
    context: int = DEFAULT_CONTEXT,
    gpu_layers: int = DEFAULT_GPU_LAYERS,
    port: int = 8080,
    host: str = "127.0.0.1",
    timeout: int = 60,
) -> subprocess.Popen[str] | None:
    """Start llama-server. Returns the process handle or None on failure.

    Args:
        model_path: Path to the GGUF model. Auto-detected if None.
        context: Context window size in tokens.
        gpu_layers: Number of layers to offload to GPU (999 = all).
        port: Server port.
        host: Server bind address.
        timeout: Seconds to wait for the server to become ready.
    """
    if is_server_running(f"http://{host}:{port}"):
        logger.info("llama.cpp server already running at %s:%d", host, port)
        return None

    server_bin = _find_server_binary()
    if server_bin is None:
        logger.error(
            "llama-server binary not found. Build llama.cpp first: "
            "python scripts/setup_qwen36_local.py --build-only"
        )
        return None

    if model_path is None:
        model_path = _find_model()
    if model_path is None:
        logger.error(
            "No GGUF model found. Download first: "
            "python scripts/setup_qwen36_local.py --download-only"
        )
        return None

    cmd = [
        str(server_bin),
        "-m", str(model_path),
        "-c", str(context),
        "--n-gpu-layers", str(gpu_layers),
        "--host", host,
        "--port", str(port),
    ]

    logger.info("Starting llama.cpp server: %s", " ".join(cmd))
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except OSError as exc:
        logger.error("Failed to start llama.cpp server: %s", exc)
        return None

    # Poll until ready
    deadline = time.monotonic() + timeout
    url = f"http://{host}:{port}"
    while time.monotonic() < deadline:
        time.sleep(0.5)
        if is_server_running(url):
            logger.info("llama.cpp server ready at %s", url)
            return proc
        if proc.poll() is not None:
            stdout, stderr = proc.communicate(timeout=5)
            logger.error("Server exited early. stdout: %s", stdout[:500])
            logger.error("stderr: %s", stderr[:500])
            return None

    logger.warning("llama.cpp server startup timed out after %d seconds", timeout)
    return proc


def stop_server(proc: subprocess.Popen[str] | None) -> bool:
    """Gracefully stop the llama.cpp server process."""
    if proc is None or proc.poll() is not None:
        return True
    try:
        proc.terminate()
        proc.wait(timeout=10)
        return True
    except subprocess.TimeoutExpired:
        logger.warning("Server did not terminate gracefully, killing")
        proc.kill()
        proc.wait(timeout=5)
        return True
    except Exception as exc:
        logger.error("Error stopping server: %s", exc)
        return False


def get_server_status(url: str = DEFAULT_URL) -> dict[str, Any]:
    """Return detailed status of the llama.cpp server."""
    status: dict[str, Any] = {
        "running": False,
        "url": url,
        "model": None,
        "version": None,
    }
    if not is_server_running(url):
        return status

    status["running"] = True

    # Try to read model info from /v1/models
    try:
        import urllib.request

        req = urllib.request.Request(f"{url}/v1/models", method="GET")  # noqa: S310
        with urllib.request.urlopen(req, timeout=2) as resp:  # noqa: S310
            import json

            data = json.loads(resp.read().decode())
            if isinstance(data, dict) and "data" in data:
                models = data["data"]
                if models:
                    status["model"] = models[0].get("id", "unknown")
            elif isinstance(data, list) and data:
                status["model"] = data[0].get("id", "unknown")
    except Exception as exc:
        logger.debug("Could not query /v1/models: %s", exc)

    # Try to read server props from /props
    try:
        import urllib.request

        req = urllib.request.Request(f"{url}/props", method="GET")  # noqa: S310
        with urllib.request.urlopen(req, timeout=2) as resp:  # noqa: S310
            import json

            data = json.loads(resp.read().decode())
            status["version"] = data.get("version", "unknown")
            status["default_generation_settings"] = data.get("default_generation_settings", {})
    except Exception as exc:
        logger.debug("Could not query /props: %s", exc)

    return status


def configure_litellm_env(url: str = DEFAULT_URL) -> None:
    """Set environment variables so LiteLLM routes to the local server.

    This must be called before LiteLLM is imported. Sets:
      - LLAMACPP_API_BASE (used by LiteLLM llamacpp provider)
      - OPENAI_API_BASE (fallback for openai/ provider prefix)
    """
    import os

    api_base = f"{url}/v1"
    os.environ.setdefault(LLAMACPP_API_BASE_ENV, api_base)
    os.environ.setdefault("OPENAI_API_BASE", api_base)
    os.environ.setdefault("OPENAI_API_KEY", "none")
    logger.debug("Configured LiteLLM env: %s=%s", LLAMACPP_API_BASE_ENV, api_base)


class LlamaCppTool(Tool):
    """Manage the local llama.cpp inference server.

    Provides status checks, start, and stop actions for the OpenAI-compatible
    server that powers local model inference.
    """

    @property
    def name(self) -> str:
        return "llamacpp"

    @property
    def description(self) -> str:
        return (
            "Manage the local llama.cpp inference server. "
            "Actions: status (check server health), start (launch server), "
            "stop (shutdown server). Use this to verify local model availability."
        )

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.LOW

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["status", "start", "stop"],
                    "description": "Action: status, start, or stop.",
                },
            },
            "required": ["action"],
        }

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        action = arguments.get("action", "status")

        if action == "status":
            status = get_server_status()
            if status["running"]:
                lines = ["llama.cpp server: RUNNING"]
                lines.append(f"  URL: {status['url']}")
                if status["model"]:
                    lines.append(f"  Model: {status['model']}")
                if status["version"]:
                    lines.append(f"  Version: {status['version']}")
                return ToolResult.ok("\n".join(lines))
            return ToolResult.ok(
                f"llama.cpp server: NOT RUNNING\n  URL: {status['url']}\n"
                "  Run 'llamacpp start' to launch."
            )

        if action == "start":
            proc = start_server()
            if proc is not None:
                return ToolResult.ok("llama.cpp server started successfully.")
            if is_server_running():
                return ToolResult.ok("llama.cpp server was already running.")
            return ToolResult.failure(
                "Failed to start llama.cpp server. "
                "Ensure the model is downloaded and llama.cpp is built."
            )

        if action == "stop":
            # We don't have the process handle in the tool context,
            # so we can't stop a server we didn't start. This is a
            # limitation; the CLI manages the lifecycle.
            return ToolResult.ok(
                "llama.cpp server stop requested. "
                "(If started by Godspeed CLI, it will stop on exit.)"
            )

        return ToolResult.failure(f"Unknown action: {action!r}")
