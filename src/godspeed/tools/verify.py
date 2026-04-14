"""Verify tool — run linter checks on files after edits.

Supports multiple languages with automatic linter detection:
- Python: ruff
- JavaScript/TypeScript: biome, eslint
- Go: go vet
- Rust: cargo check (via clippy)
- C/C++: clang-tidy
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any

from godspeed.tools.base import RiskLevel, Tool, ToolContext, ToolResult
from godspeed.tools.path_utils import resolve_tool_path

logger = logging.getLogger(__name__)

# Languages where auto-fix is safe and deterministic
_FIXABLE_LANGUAGES: frozenset[str] = frozenset({"python", "javascript", "typescript"})

# Timeout for linter subprocess
VERIFY_TIMEOUT = 30

# Extension → verifier function mapping
_EXTENSION_MAP: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".c": "c_cpp",
    ".cpp": "c_cpp",
    ".h": "c_cpp",
    ".hpp": "c_cpp",
}


class VerifyTool(Tool):
    """Run linter verification on a file.

    Supports Python (ruff), JS/TS (biome/eslint), Go (go vet),
    Rust (cargo check), C/C++ (clang-tidy). Returns lint errors/warnings
    so the agent can self-correct. Gracefully returns success for
    unsupported file types or missing linters.
    """

    @property
    def name(self) -> str:
        return "verify"

    @property
    def description(self) -> str:
        return (
            "Run linter checks on a file to catch syntax errors and style issues. "
            "Supports Python (ruff), JS/TS (biome/eslint), Go (go vet), "
            "Rust (cargo check), C/C++ (clang-tidy). Returns clean or error details.\n\n"
            "Example: verify(file_path='src/app.py')\n"
            "Example: verify(file_path='index.ts')"
        )

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.READ_ONLY

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to verify (relative to project root)",
                    "examples": ["src/app.py", "index.ts", "main.go"],
                },
            },
            "required": ["file_path"],
        }

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        file_path_str = arguments.get("file_path", "")

        if not isinstance(file_path_str, str) or not file_path_str:
            return ToolResult.failure("file_path must be a non-empty string")

        try:
            resolved = resolve_tool_path(file_path_str, context.cwd)
        except ValueError as exc:
            return ToolResult.failure(str(exc))

        if not resolved.exists():
            return ToolResult.failure(f"File not found: {file_path_str}")

        suffix = resolved.suffix.lower()
        lang = _EXTENSION_MAP.get(suffix)

        if lang == "python":
            return _verify_python(resolved, file_path_str)
        if lang in ("javascript", "typescript"):
            return _verify_js_ts(resolved, file_path_str, context.cwd)
        if lang == "go":
            return _verify_go(resolved, file_path_str)
        if lang == "rust":
            return _verify_rust(resolved, file_path_str)
        if lang == "c_cpp":
            return _verify_c_cpp(resolved, file_path_str)

        return ToolResult.success(
            f"No linter configured for {suffix} files. Skipping verification."
        )


def _run_linter(cmd: list[str], display_path: str, linter_name: str) -> ToolResult:
    """Run a linter command and return a ToolResult."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=VERIFY_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return ToolResult.failure(f"Verification timed out after {VERIFY_TIMEOUT}s: {display_path}")
    except OSError as exc:
        return ToolResult.failure(f"Failed to run {linter_name}: {exc}")

    if result.returncode == 0:
        return ToolResult.success(f"Verification passed ({linter_name}): {display_path}")

    output = result.stdout.strip() or result.stderr.strip()
    return ToolResult.success(f"Lint issues in {display_path} ({linter_name}):\n{output}")


def _verify_python(resolved: Path, display_path: str, fix_mode: bool = False) -> ToolResult:
    """Run ruff check on a Python file.

    Args:
        resolved: Absolute path to the file.
        display_path: Relative path for user-facing messages.
        fix_mode: When True, run ``ruff check --fix`` instead of ``--no-fix``.
    """
    ruff_bin = shutil.which("ruff")
    if ruff_bin is None:
        return ToolResult.success(
            "ruff not found — skipping verification. Install with: pip install ruff"
        )
    fix_flag = "--fix" if fix_mode else "--no-fix"
    return _run_linter(
        [ruff_bin, "check", "--select=E,W,F", fix_flag, str(resolved)],
        display_path,
        "ruff",
    )


def _verify_js_ts(resolved: Path, display_path: str, cwd: Path) -> ToolResult:
    """Run biome or eslint on a JS/TS file."""
    # Prefer biome (faster, no config needed)
    biome_bin = shutil.which("biome")
    if biome_bin is not None:
        return _run_linter(
            [biome_bin, "check", "--no-errors-on-unmatched", str(resolved)],
            display_path,
            "biome",
        )

    # Fall back to eslint
    eslint_bin = shutil.which("eslint")
    if eslint_bin is not None:
        return _run_linter(
            [eslint_bin, "--no-fix", str(resolved)],
            display_path,
            "eslint",
        )

    # Try npx eslint as last resort if node_modules exists
    npx_bin = shutil.which("npx")
    if npx_bin is not None and (cwd / "node_modules").is_dir():
        return _run_linter(
            [npx_bin, "eslint", "--no-fix", str(resolved)],
            display_path,
            "eslint (npx)",
        )

    return ToolResult.success(f"No JS/TS linter found for {display_path}. Install biome or eslint.")


def _verify_go(resolved: Path, display_path: str) -> ToolResult:
    """Run go vet on a Go file."""
    go_bin = shutil.which("go")
    if go_bin is None:
        return ToolResult.success("go not found — skipping verification.")
    return _run_linter(
        [go_bin, "vet", str(resolved)],
        display_path,
        "go vet",
    )


def _verify_rust(resolved: Path, display_path: str) -> ToolResult:
    """Run cargo check for Rust files.

    Rust's linter (clippy) works on crate level, not individual files,
    so we run cargo check from the file's directory to catch compile errors.
    """
    cargo_bin = shutil.which("cargo")
    if cargo_bin is None:
        return ToolResult.success("cargo not found — skipping verification.")

    # Find the nearest Cargo.toml
    cargo_dir = resolved.parent
    while cargo_dir != cargo_dir.parent:
        if (cargo_dir / "Cargo.toml").exists():
            break
        cargo_dir = cargo_dir.parent
    else:
        return ToolResult.success(f"No Cargo.toml found for {display_path}. Skipping.")

    try:
        result = subprocess.run(
            [cargo_bin, "check", "--message-format=short"],
            capture_output=True,
            text=True,
            timeout=VERIFY_TIMEOUT,
            cwd=str(cargo_dir),
        )
    except subprocess.TimeoutExpired:
        return ToolResult.failure(f"Verification timed out after {VERIFY_TIMEOUT}s: {display_path}")
    except OSError as exc:
        return ToolResult.failure(f"Failed to run cargo check: {exc}")

    if result.returncode == 0:
        return ToolResult.success(f"Verification passed (cargo check): {display_path}")

    output = result.stderr.strip() or result.stdout.strip()
    return ToolResult.success(f"Build issues in {display_path} (cargo check):\n{output}")


def _verify_c_cpp(resolved: Path, display_path: str) -> ToolResult:
    """Run clang-tidy on a C/C++ file."""
    clang_tidy = shutil.which("clang-tidy")
    if clang_tidy is None:
        return ToolResult.success("clang-tidy not found — skipping verification.")
    return _run_linter(
        [clang_tidy, str(resolved), "--quiet"],
        display_path,
        "clang-tidy",
    )


def _has_lint_issues(result: ToolResult) -> bool:
    """Return True if a verify ToolResult indicates lint issues (not a clean pass)."""
    if result.is_error:
        return True
    output_lower = result.output.lower()
    return "lint issues" in output_lower or "build issues" in output_lower


def _verify_with_retry(
    resolved: Path,
    display_path: str,
    lang: str,
    cwd: Path,
    max_retries: int = 3,
) -> ToolResult:
    """Run verify, auto-fix, and re-verify in a retry loop.

    Only applies auto-fix for Python and JS/TS where fix tools are deterministic.
    Go, Rust, and C/C++ skip the retry loop and return one-shot results.

    Args:
        resolved: Absolute path to the file.
        display_path: Relative path for user-facing messages.
        lang: Language key from ``_EXTENSION_MAP``.
        cwd: Project working directory (needed for JS/TS linters).
        max_retries: Maximum fix-then-recheck cycles. 0 means one-shot (no fix).

    Returns:
        ToolResult with summary of fixes applied and remaining issues.
    """
    # One-shot for non-fixable languages or when retries are disabled
    if lang not in _FIXABLE_LANGUAGES or max_retries <= 0:
        return _one_shot_verify(resolved, display_path, lang, cwd)

    # Initial check
    result = _one_shot_verify(resolved, display_path, lang, cwd)
    if not _has_lint_issues(result):
        return ToolResult.success(f"Verification passed: {display_path}")

    total_fixed = 0
    for attempt in range(max_retries):
        # Run fix pass
        _run_fix(resolved, display_path, lang, cwd)

        # Re-check
        recheck = _one_shot_verify(resolved, display_path, lang, cwd)
        if not _has_lint_issues(recheck):
            total_fixed += 1  # At least one round of fixes applied
            return ToolResult.success(
                f"Auto-fixed {attempt + 1} round(s) of issues, 0 remaining: {display_path}"
            )
        total_fixed += 1

    # Exhausted retries — report remaining issues
    final = _one_shot_verify(resolved, display_path, lang, cwd)
    remaining_output = final.output if not final.is_error else (final.error or "")
    return ToolResult.success(
        f"Auto-fixed {total_fixed} round(s) of issues, "
        f"some remaining: {display_path}\n{remaining_output}"
    )


def _one_shot_verify(resolved: Path, display_path: str, lang: str, cwd: Path) -> ToolResult:
    """Run a single verify pass (no fix) for the given language."""
    if lang == "python":
        return _verify_python(resolved, display_path, fix_mode=False)
    if lang in ("javascript", "typescript"):
        return _verify_js_ts(resolved, display_path, cwd)
    if lang == "go":
        return _verify_go(resolved, display_path)
    if lang == "rust":
        return _verify_rust(resolved, display_path)
    if lang == "c_cpp":
        return _verify_c_cpp(resolved, display_path)
    return ToolResult.success(f"No linter configured for {lang}. Skipping verification.")


def _run_fix(resolved: Path, display_path: str, lang: str, cwd: Path) -> None:
    """Run a fix pass for fixable languages. Errors are silently ignored."""
    import contextlib

    if lang == "python":
        _verify_python(resolved, display_path, fix_mode=True)
    elif lang in ("javascript", "typescript"):
        # biome: `biome check --write`, eslint: `eslint --fix`
        biome_bin = shutil.which("biome")
        if biome_bin is not None:
            with contextlib.suppress(subprocess.TimeoutExpired, OSError):
                subprocess.run(
                    [biome_bin, "check", "--write", str(resolved)],
                    capture_output=True,
                    text=True,
                    timeout=VERIFY_TIMEOUT,
                )
            return
        eslint_bin = shutil.which("eslint")
        if eslint_bin is not None:
            with contextlib.suppress(subprocess.TimeoutExpired, OSError):
                subprocess.run(
                    [eslint_bin, "--fix", str(resolved)],
                    capture_output=True,
                    text=True,
                    timeout=VERIFY_TIMEOUT,
                )
