"""Run a single SWE-Bench patch through the local Docker harness.

Returns (resolved: bool, test_output: str). Used as the oracle signal
for the verify-then-retry loop in run.py and the agent-in-loop
`swebench_verify_patch` tool.

Two execution paths:

1. **WSL path** (Windows) - wraps `python3 -m swebench.harness.run_evaluation`
   inside `wsl -d Ubuntu -- bash -lc '...'`. Windows paths are translated
   to `/mnt/<drive>/...` for the WSL command. Swebench is installed inside
   the Ubuntu distro via `pip3 install --break-system-packages --user swebench`.

2. **Native path** (Linux, CI) - calls `python3 -m swebench.harness.run_evaluation`
   directly via `subprocess.run`. Swebench must be installed in the calling
   env (`pip install swebench`). No path translation.

Autodetection: native if `sys.platform != "win32"`. Override with
`GODSPEED_SWEBENCH_WSL=0` (force native) or `=1` (force WSL, e.g. for
debugging the WSL path from a Linux container).

Typical use from run.py (imported directly):

    from experiments.swebench_lite.verify_patch import verify_patch
    resolved, test_output = verify_patch(
        instance_id="sqlfluff__sqlfluff-2419",
        model_name="nvidia_nim/moonshotai/kimi-k2.5",
        model_patch=patch_str,
        workdir=Path("experiments/swebench_lite"),
    )

Standalone:

    python experiments/swebench_lite/verify_patch.py \
        --instance sqlfluff__sqlfluff-2419 \
        --model nvidia_nim/moonshotai/kimi-k2.5 \
        --patch-from experiments/swebench_lite/predictions_e1_kimi.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# WSL command prefix. Uses the default Ubuntu distro.
WSL_CMD = ["wsl", "-d", "Ubuntu", "--", "bash", "-lc"]


def _use_wsl() -> bool:
    """Decide whether to run the harness via WSL or natively.

    Env override: ``GODSPEED_SWEBENCH_WSL`` in ``{"0","false","no"}`` forces
    native; anything truthy forces WSL. Unset/empty autodetects on platform.
    """
    override = os.environ.get("GODSPEED_SWEBENCH_WSL", "").strip().lower()
    if override in ("0", "false", "no"):
        return False
    if override in ("1", "true", "yes"):
        return True
    return sys.platform == "win32"


def _wsl_run(bash_cmd: str, timeout: int = 900) -> subprocess.CompletedProcess[str]:
    """Run a bash command inside WSL Ubuntu with sensible defaults."""
    return subprocess.run(
        [*WSL_CMD, bash_cmd],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _native_run(bash_cmd: str, timeout: int = 900) -> subprocess.CompletedProcess[str]:
    """Run a bash command natively (Linux/CI path)."""
    return subprocess.run(
        ["bash", "-lc", bash_cmd],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _windows_to_wsl(p: Path) -> str:
    """Convert a Windows path like C:\\Users\\x to /mnt/c/Users/x for WSL."""
    parts = list(p.resolve().parts)
    if len(parts) == 0:
        return str(p)
    drive = parts[0].rstrip(":\\").lower()
    return "/mnt/" + drive + "/" + "/".join(parts[1:]).replace("\\", "/")


def _harness_cmd(workdir_str: str, preds_str: str, instance_id: str, run_id: str) -> str:
    """Compose the bash command to run the swebench harness.

    ``workdir_str`` and ``preds_str`` are already in the format the shell
    will consume (POSIX for WSL/native, Windows-translated for WSL).
    """
    return (
        f"cd '{workdir_str}' && "
        f"python3 -m swebench.harness.run_evaluation "
        f"--predictions_path '{preds_str}' "
        f"--dataset_name princeton-nlp/SWE-bench_Lite "
        f"--split dev "
        f"--instance_ids {instance_id} "
        f"--max_workers 1 "
        f"--run_id {run_id} "
        f"--cache_level instance"
    )


def verify_patch(
    instance_id: str,
    model_name: str,
    model_patch: str,
    workdir: Path,
    timeout_s: int = 900,
) -> tuple[bool, str]:
    """Run the swebench harness on a single patch via local Docker.

    Returns ``(resolved, test_output)``. ``resolved`` is ``True`` iff the
    harness reports the instance as resolved. ``test_output`` is the raw
    test_output.txt contents if the harness produced one, or a short error
    summary if the harness itself failed.
    """
    if not model_patch.strip():
        return False, "(empty patch - nothing to verify)"

    workdir = workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    # Unique run id — hash of (instance + patch) so repeated calls on the
    # same content reuse harness artifacts.
    digest = hashlib.sha1(
        (instance_id + "::" + model_patch).encode("utf-8"), usedforsecurity=False
    ).hexdigest()[:12]
    run_id = f"verify_{instance_id.replace('/', '_')}_{digest}"

    # Write single-instance predictions file
    preds_path = workdir / f".verify_{digest}.jsonl"
    preds_path.write_text(
        json.dumps(
            {
                "instance_id": instance_id,
                "model_name_or_path": model_name,
                "model_patch": model_patch,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    if _use_wsl():
        workdir_str = _windows_to_wsl(workdir)
        preds_str = _windows_to_wsl(preds_path)
        bash_cmd = _harness_cmd(workdir_str, preds_str, instance_id, run_id)
        logger.info("verify harness (wsl): %s (timeout %ds)", instance_id, timeout_s)
        result = _wsl_run(bash_cmd, timeout=timeout_s)
    else:
        bash_cmd = _harness_cmd(str(workdir), str(preds_path), instance_id, run_id)
        logger.info("verify harness (native): %s (timeout %ds)", instance_id, timeout_s)
        result = _native_run(bash_cmd, timeout=timeout_s)

    # Expected report path (written to cwd by the harness).
    # Normalize the model name the same way the harness does: "/" -> "__"
    model_norm = model_name.replace("/", "__")
    report_path = workdir / f"{model_norm}.{run_id}.json"
    if not report_path.is_file():
        logger.warning(
            "verify: report not found at %s - harness likely failed. stderr tail:\n%s",
            report_path,
            result.stderr[-500:],
        )
        return False, f"(harness failed)\n{result.stderr[-1000:]}"

    report = json.loads(report_path.read_text(encoding="utf-8"))
    resolved = instance_id in report.get("resolved_ids", [])

    # Fetch test output for the agent's retry prompt context.
    # swebench writes logs/run_evaluation/<run_id>/<model>/<instance>/test_output.txt
    log_rel = Path("logs/run_evaluation") / run_id / model_norm / instance_id / "test_output.txt"
    log_path = workdir / log_rel
    if log_path.is_file():
        test_output = log_path.read_text(encoding="utf-8", errors="replace")
    else:
        # Fallback: look in the whole logs tree (swebench's exact layout
        # has varied across versions).
        matches = (
            list((workdir / "logs").rglob("test_output.txt")) if (workdir / "logs").is_dir() else []
        )
        test_output = (
            matches[-1].read_text(encoding="utf-8", errors="replace")
            if matches
            else "(no test_output.txt found)"
        )

    return resolved, test_output


def _main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--patch-from",
        type=Path,
        required=True,
        help="A predictions.jsonl file - we pick the row whose instance_id matches --instance",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path("experiments/swebench_lite"),
    )
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()

    for line in args.patch_from.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["instance_id"] == args.instance:
            patch = row["model_patch"]
            break
    else:
        raise SystemExit(f"instance {args.instance} not in {args.patch_from}")

    resolved, test_output = verify_patch(
        instance_id=args.instance,
        model_name=args.model,
        model_patch=patch,
        workdir=args.workdir,
        timeout_s=args.timeout,
    )
    print(f"resolved: {resolved}")
    print()
    print("--- test_output (tail) ---")
    print(test_output[-2000:])
    return 0 if resolved else 1


if __name__ == "__main__":
    sys.exit(_main())
