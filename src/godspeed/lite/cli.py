"""Godspeed Lite CLI — the fast coding agent.

Usage:
    godspeed-lite "fix the IndexError in data_loader.py"
    godspeed-lite --mode deep "debug this race condition"
    godspeed-lite --mode rush "add a docstring to main.py"
    godspeed-lite --model openai/gpt-5.2 "add dark mode toggle"
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from godspeed.lite.agent import MODES, GodspeedLite

logger = logging.getLogger("godspeed.lite")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Godspeed Lite — fast bash-only coding agent",
        epilog="Soli Deo Gloria.",
    )
    parser.add_argument(
        "task",
        nargs="*",
        default=None,
        help="Task description. Pass '-' to read from stdin.",
    )
    parser.add_argument(
        "--mode",
        choices=sorted(MODES),
        default="smart",
        help="Agent mode: smart (default), rush (fast/cheap), deep (thorough)",
    )
    parser.add_argument("--model", default=None, help="Override the default model")
    parser.add_argument("--workdir", default=None, help="Working directory (default: cwd)")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--step-timeout", type=int, default=None)
    parser.add_argument("--budget-after", type=int, default=None)
    parser.add_argument(
        "--single-shot",
        action="store_true",
        help="Single LLM call mode (works on tight rate limits)",
    )
    parser.add_argument("--no-verify", action="store_true", help="Skip post-hoc verification")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    task = _read_task(args)
    if not task:
        parser.error("No task provided. Specify a task or pipe input.")

    workdir = Path(args.workdir) if args.workdir else Path.cwd()
    agent = GodspeedLite(
        mode=args.mode,
        workdir=workdir,
        model=args.model,
        max_steps=1 if args.single_shot else args.max_steps,
        step_timeout=args.step_timeout,
        budget_after=args.budget_after,
    )

    logger.info(
        "mode=%s model=%s workdir=%s",
        args.mode,
        agent._cfg.model,
        workdir,
    )

    patch = asyncio.run(agent.run(task))

    if patch.strip():
        print(patch)  # noqa: T201
    else:
        logger.warning("No patch produced — agent did not make changes")
        print("(no changes)", file=sys.stderr)  # noqa: T201

    logger.info(
        "done: steps=%d cost=$%.4f models=%s",
        agent.steps_taken,
        agent.cost_usd,
        ", ".join(set(agent.models_used)),
    )
    return 0 if patch.strip() else 1


def _read_task(args: argparse.Namespace) -> str:
    if args.task:
        task = " ".join(args.task)
        if task == "-":
            return sys.stdin.read()
        return task
    if not sys.stdin.isatty():
        return sys.stdin.read()
    return ""


if __name__ == "__main__":
    sys.exit(main())
