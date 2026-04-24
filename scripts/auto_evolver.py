#!/usr/bin/env python
"""Godspeed Auto-Evolver 24/7 - Continuous improvement loop."""

import asyncio
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# Setup logging - Windows compatible
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("godspeed_evolver.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# Evolution tasks - Windows compatible commands
EVOLUTION_TASKS = [
    {
        "name": "Lint & Format",
        "command": ["python", "-m", "ruff", "check", "src/", "--fix"],
        "interval_hours": 1,
        "priority": "high",
    },
    {
        "name": "Run Tests",
        "command": ["python", "-m", "pytest", "tests/", "-x", "--tb=short", "-q"],
        "interval_hours": 2,
        "priority": "critical",
    },
    {
        "name": "Type Check",
        "command": ["python", "-m", "mypy", "src/godspeed", "--ignore-missing-imports"],
        "interval_hours": 4,
        "priority": "medium",
    },
    {
        "name": "Pull Latest",
        "command": ["git", "pull", "origin", "main"],
        "interval_hours": 6,
        "priority": "high",
    },
    {
        "name": "Clean Cache",
        "command": ["python", "-c", "import shutil; [shutil.rmtree(p, ignore_errors=True) for p in Path('.').rglob('__pycache__')]"],
        "interval_hours": 12,
        "priority": "low",
    },
]


class AutoEvolver:
    def __init__(self, project_dir: Path):
        self.project_dir = project_dir
        self.last_run: dict[str, datetime] = {}
        self.stats = {"runs": 0, "success": 0, "failures": 0}

    async def run_command(self, cmd: list[str], timeout: int = 300) -> tuple[bool, str]:
        """Run a shell command and return success status + output."""
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=self.project_dir,
            )
            try:
                output, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
                success = proc.returncode == 0
                return success, output.decode("utf-8", errors="replace")
            except asyncio.TimeoutError:
                proc.kill()
                return False, f"Timeout after {timeout}s"
        except Exception as e:
            return False, str(e)

    async def should_run(self, task: dict) -> bool:
        """Check if task should run based on interval."""
        name = task["name"]
        interval = timedelta(hours=task["interval_hours"])
        last = self.last_run.get(name)
        if last is None:
            return True
        return datetime.now() - last > interval

    async def run_task(self, task: dict) -> tuple[bool, str]:
        """Execute a single evolution task."""
        name = task["name"]
        cmd = task["command"]
        timeout = min(task["interval_hours"] * 1800, 3600)

        logger.info(f"[RUN] {name}")
        start = time.time()
        success, output = await self.run_command(cmd, timeout=timeout)
        elapsed = time.time() - start

        self.last_run[name] = datetime.now()
        self.stats["runs"] += 1
        if success:
            self.stats["success"] += 1
            logger.info(f"[OK] {name} completed in {elapsed:.1f}s")
        else:
            self.stats["failures"] += 1
            logger.warning(f"[FAIL] {name}: {output[:200]}")

        return success, output

    async def run_evolution_cycle(self) -> None:
        """Run one complete evolution cycle."""
        logger.info("=" * 50)
        logger.info(f"Evolution cycle {self.stats['runs'] + 1} starting")
        logger.info("=" * 50)

        for task in EVOLUTION_TASKS:
            if await self.should_run(task):
                await self.run_task(task)
                await asyncio.sleep(1)

        logger.info(f"Stats: {self.stats}")

    async def run_forever(self, cycle_interval_hours: float = 1.0) -> None:
        """Run the evolver in an infinite loop."""
        logger.info(f"Godspeed Auto-Evolver starting (cycle every {cycle_interval_hours}h)")
        logger.info(f"Project: {self.project_dir}")

        env_file = Path.home() / ".godspeed" / ".env.local"
        if env_file.exists():
            logger.info("API key auto-loaded from ~/.godspeed/.env.local")

        while True:
            try:
                await self.run_evolution_cycle()
            except Exception as e:
                logger.error(f"Cycle failed: {e}")
                self.stats["failures"] += 1

            await asyncio.sleep(cycle_interval_hours * 3600)


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Godspeed Auto-Evolver 24/7")
    parser.add_argument("--project-dir", type=Path, default=Path.cwd())
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    evolver = AutoEvolver(args.project_dir)

    if args.once:
        await evolver.run_evolution_cycle()
    else:
        await evolver.run_forever(args.interval)


if __name__ == "__main__":
    asyncio.run(main())