"""Windows setup script for Qwen3.6-27B local inference with llama.cpp.

Downloads the UD-Q4_K_XL GGUF, builds llama.cpp with CUDA support,
and launches an OpenAI-compatible server optimized for RTX 5070 Ti.

Usage:
    python scripts/setup_qwen36_local.py [--download-only|--build-only|--launch-only]

Prerequisites (Windows):
    - CUDA 12.x or 13.1 toolkit (NOT 13.2)
    - Git for Windows
    - CMake >= 3.20
    - Visual Studio 2022 Build Tools (MSVC)
    - 25GB free disk space
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# Model configuration
MODEL_REPO = "unsloth/Qwen3.6-27B-GGUF"
MODEL_FILE = "Qwen3.6-27B-UD-Q4_K_XL.gguf"
MODEL_SIZE_GB = 17
DEFAULT_MODELS_DIR = Path.home() / ".llamacpp" / "models"
DEFAULT_LLAMACPP_DIR = Path.home() / "llama.cpp"
DEFAULT_PORT = 8080
DEFAULT_CONTEXT = 32768

# RTX 5070 Ti optimization flags
GPU_LAYERS = 999  # offload all layers; llama.cpp auto-manages VRAM
HOST = "127.0.0.1"  # local only for security


def _has_nvidia_gpu() -> bool:
    """Check if an NVIDIA GPU is available."""
    return shutil.which("nvidia-smi") is not None


def _get_cuda_version() -> str | None:
    """Get installed CUDA version from nvcc."""
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        return None
    try:
        result = subprocess.run(
            [nvcc, "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
        for line in result.stdout.splitlines():
            if "release" in line:
                return line.strip()
        return result.stdout.strip()[:200]
    except Exception as exc:
        logger.debug("nvcc check failed: %s", exc)
        return None


def _check_disk_space(path: Path, required_gb: int) -> bool:
    """Check if path has at least required_gb free."""
    try:
        if sys.platform == "win32":
            import ctypes

            free_bytes = ctypes.c_ulonglong(0)
            ctypes.windll.kernel32.GetDiskFreeSpaceExW(  # type: ignore[attr-defined]
                str(path), None, None, ctypes.byref(free_bytes)
            )
            free_gb = free_bytes.value / (1024**3)
        else:
            stat = os.statvfs(path)
            free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)

        if free_gb < required_gb:
            logger.error(
                "Insufficient disk space at %s: %.1f GB free, need %d GB",
                path,
                free_gb,
                required_gb,
            )
            return False
        logger.info("Disk space OK: %.1f GB free at %s", free_gb, path)
        return True
    except Exception as exc:
        logger.warning("Could not check disk space: %s", exc)
        return True  # permissive


def _run_subprocess(
    cmd: list[str],
    cwd: Path | None = None,
    timeout: int = 300,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run a subprocess with logging."""
    logger.info("Running: %s", " ".join(cmd))
    return subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=check,
    )


def download_model(models_dir: Path, *, dry_run: bool = False) -> Path | None:
    """Download the Qwen3.6-27B GGUF from HuggingFace."""
    target = models_dir / MODEL_FILE
    if target.exists():
        size_gb = target.stat().st_size / (1024**3)
        if size_gb >= MODEL_SIZE_GB * 0.95:
            logger.info("Model already downloaded: %s (%.1f GB)", target, size_gb)
            return target
        logger.warning("Incomplete download detected (%.1f GB), re-downloading", size_gb)
        target.unlink()

    if not _check_disk_space(models_dir, MODEL_SIZE_GB + 5):
        return None

    models_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading %s/%s...", MODEL_REPO, MODEL_FILE)
    if dry_run:
        logger.info("[DRY-RUN] Would download model to %s", target)
        return target

    # Prefer huggingface-cli if available
    hf_cli = shutil.which("huggingface-cli")
    if hf_cli is not None:
        try:
            _run_subprocess(
                [
                    sys.executable,
                    "-m",
                    "huggingface_hub",
                    "download",
                    MODEL_REPO,
                    MODEL_FILE,
                    "--local-dir",
                    str(models_dir),
                ],
                timeout=1800,
            )
            if target.exists():
                logger.info("Download complete: %s", target)
                return target
        except subprocess.CalledProcessError as exc:
            logger.warning("huggingface-cli download failed: %s", exc.stderr[:500])

    # Fallback: curl/wget direct download
    url = f"https://huggingface.co/{MODEL_REPO}/resolve/main/{MODEL_FILE}"
    logger.info("Trying direct download from %s", url)
    try:
        if shutil.which("curl"):
            _run_subprocess(
                ["curl", "-L", "-o", str(target), url],
                timeout=1800,
                check=False,
            )
        elif shutil.which("wget"):
            _run_subprocess(
                ["wget", "-O", str(target), url],
                timeout=1800,
                check=False,
            )
        else:
            logger.error(
                "No download tool found. Install huggingface_hub: pip install huggingface_hub"
            )
            return None

        if target.exists() and target.stat().st_size > 1024**3:
            logger.info("Download complete: %s", target)
            return target
    except Exception as exc:
        logger.error("Direct download failed: %s", exc)

    return None


def build_llamacpp(build_dir: Path, *, dry_run: bool = False) -> Path | None:
    """Build llama.cpp with CUDA support. Returns path to llama-server binary."""
    server_bin = build_dir / "bin" / "llama-server"
    if sys.platform == "win32":
        server_bin = build_dir / "bin" / "Release" / "llama-server.exe"

    if server_bin.exists():
        logger.info("llama-server already built: %s", server_bin)
        return server_bin

    if dry_run:
        logger.info("[DRY-RUN] Would build llama.cpp at %s", build_dir)
        return server_bin

    logger.info("Cloning llama.cpp...")
    llamacpp_dir = build_dir.parent
    if not llamacpp_dir.exists():
        try:
            _run_subprocess(
                ["git", "clone", "https://github.com/ggerganov/llama.cpp", str(llamacpp_dir)],
                timeout=120,
            )
        except subprocess.CalledProcessError as exc:
            logger.error("Git clone failed: %s", exc.stderr[:500])
            return None

    logger.info("Building llama.cpp with CUDA...")
    try:
        # Configure
        cmake_args = [
            "cmake",
            "-B",
            str(build_dir),
            "-DGGML_CUDA=ON",
            "-DLLAMA_BUILD_SERVER=ON",
        ]
        if sys.platform == "win32":
            cmake_args.extend(["-G", "Visual Studio 17 2022", "-A", "x64"])

        result = subprocess.run(
            cmake_args,
            cwd=llamacpp_dir,
            capture_output=True,
            text=True,
            check=False,
        )
        if "CUDA found" not in result.stdout and "CUDA found" not in result.stderr:
            logger.warning("CUDA may not have been detected. Check cmake output.")
            logger.warning("stdout: %s", result.stdout[:1000])
            logger.warning("stderr: %s", result.stderr[:1000])

        # Build
        build_cmd = ["cmake", "--build", str(build_dir), "--config", "Release"]
        if sys.platform != "win32":
            build_cmd.extend(["-j", str(os.cpu_count() or 4)])

        _run_subprocess(build_cmd, cwd=llamacpp_dir, timeout=600, check=False)

        if server_bin.exists():
            logger.info("Build successful: %s", server_bin)
            return server_bin
        # Try alternative path
        alt_bin = build_dir / "llama-server"
        if sys.platform == "win32":
            alt_bin = build_dir / "Release" / "llama-server.exe"
        if alt_bin.exists():
            logger.info("Build successful: %s", alt_bin)
            return alt_bin

        logger.error("Build completed but llama-server binary not found")
        return None
    except Exception as exc:
        logger.error("Build failed: %s", exc)
        return None


def launch_server(
    server_bin: Path,
    model_path: Path,
    *,
    port: int = DEFAULT_PORT,
    context: int = DEFAULT_CONTEXT,
    dry_run: bool = False,
) -> subprocess.Popen[str] | None:
    """Launch llama-server. Returns the process handle."""
    cmd = [
        str(server_bin),
        "-m",
        str(model_path),
        "-c",
        str(context),
        "--n-gpu-layers",
        str(GPU_LAYERS),
        "--host",
        HOST,
        "--port",
        str(port),
    ]

    logger.info("Launching llama-server:")
    logger.info("  Model: %s", model_path)
    logger.info("  Context: %d", context)
    logger.info("  GPU layers: %d (auto)", GPU_LAYERS)
    logger.info("  Endpoint: http://%s:%d/v1", HOST, port)

    if dry_run:
        logger.info("[DRY-RUN] Would run: %s", " ".join(cmd))
        return None

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        # Brief wait to catch immediate failures
        time.sleep(2)
        if proc.poll() is not None:
            stdout, stderr = proc.communicate(timeout=5)
            logger.error("Server exited immediately. stdout: %s", stdout[:500])
            logger.error("stderr: %s", stderr[:500])
            return None
        logger.info("Server started (PID %d)", proc.pid)
        return proc
    except Exception as exc:
        logger.error("Failed to start server: %s", exc)
        return None


def wait_for_server(port: int = DEFAULT_PORT, timeout: int = 60) -> bool:
    """Poll the health endpoint until the server is ready."""
    import urllib.request

    url = f"http://127.0.0.1:{port}/v1/models"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            req = urllib.request.Request(url, method="GET")  # noqa: S310
            with urllib.request.urlopen(req, timeout=2):  # noqa: S310
                logger.info("Server ready at %s", url)
                return True
        except Exception:
            time.sleep(1)
    logger.error("Server did not become ready within %d seconds", timeout)
    return False


def main() -> int:
    """Run the full setup pipeline."""
    parser = argparse.ArgumentParser(description="Setup Qwen3.6-27B local inference")
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--llamacpp-dir", type=Path, default=DEFAULT_LLAMACPP_DIR)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--context", type=int, default=DEFAULT_CONTEXT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--launch-only", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info("=" * 60)
    logger.info("Qwen3.6-27B Local Inference Setup")
    logger.info("=" * 60)

    # Preflight
    if not _has_nvidia_gpu():
        logger.error("No NVIDIA GPU detected. This setup requires CUDA.")
        return 1

    cuda_ver = _get_cuda_version()
    if cuda_ver:
        logger.info("CUDA: %s", cuda_ver)
        if "13.2" in cuda_ver:
            logger.error(
                "CUDA 13.2 detected — outputs will be gibberish. Downgrade to 12.x or 13.1."
            )
            return 1
    else:
        logger.warning("Could not detect CUDA version")

    # Determine phases to run
    run_download = args.download_only or not (args.build_only or args.launch_only)
    run_build = args.build_only or not (args.download_only or args.launch_only)
    run_launch = args.launch_only or not (args.download_only or args.build_only)

    model_path: Path | None = None
    server_bin: Path | None = None

    # Phase 1: Download
    if run_download:
        logger.info("\n--- Phase 1: Download Model ---")
        model_path = download_model(args.models_dir, dry_run=args.dry_run)
        if model_path is None and not args.dry_run:
            logger.error("Model download failed")
            return 1

    # Phase 2: Build
    if run_build:
        logger.info("\n--- Phase 2: Build llama.cpp ---")
        build_dir = args.llamacpp_dir / "build"
        server_bin = build_llamacpp(build_dir, dry_run=args.dry_run)
        if server_bin is None and not args.dry_run:
            logger.error("Build failed")
            return 1

    # Phase 3: Launch
    if run_launch:
        logger.info("\n--- Phase 3: Launch Server ---")
        if model_path is None:
            model_path = args.models_dir / MODEL_FILE
        if server_bin is None:
            server_bin = args.llamacpp_dir / "build" / "bin" / "Release" / "llama-server.exe"
            if sys.platform != "win32":
                server_bin = args.llamacpp_dir / "build" / "bin" / "llama-server"

        if not model_path.exists() and not args.dry_run:
            logger.error("Model not found: %s", model_path)
            return 1
        if not server_bin.exists() and not args.dry_run:
            logger.error("Server binary not found: %s", server_bin)
            return 1

        proc = launch_server(
            server_bin,
            model_path,
            port=args.port,
            context=args.context,
            dry_run=args.dry_run,
        )
        if proc is None and not args.dry_run:
            return 1

        if not args.dry_run and proc is not None:
            if wait_for_server(args.port):
                logger.info("\nServer is running. Press Ctrl+C to stop.")
                try:
                    proc.wait()
                except KeyboardInterrupt:
                    logger.info("\nStopping server...")
                    proc.terminate()
                    proc.wait(timeout=10)
            else:
                proc.terminate()
                return 1

    logger.info("\nSetup complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
