#!/usr/bin/env python
"""Godspeed Auto-Evolver 24/7 - Infinite loop for continuous improvement."""

import asyncio
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# FIXED: Get project directory properly
PROJECT_DIR = Path(__file__).parent.parent.resolve()
SCRIPT_DIR = Path(__file__).parent.resolve()

def load_env():
    env_file = Path.home() / ".godspeed" / ".env.local"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and "=" in line and not line.startswith("#"):
                    key, val = line.split("=", 1)
                    os.environ.setdefault(key, val)

load_env()

log_file = PROJECT_DIR / "godspeed_evolver.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

CYCLE_INTERVAL_HOURS = 1.0

TASKS = [
    {"name": "Lint", "cmd": [sys.executable, "-m", "ruff", "check", "src/", "--fix"], "interval": 1},
    {"name": "Tests", "cmd": [sys.executable, "-m", "pytest", "tests/", "-x", "-q", "--ignore=tests/test_background.py"], "interval": 2},
    {"name": "TypeCheck", "cmd": [sys.executable, "-m", "mypy", "src/godspeed", "--ignore-missing-imports"], "interval": 4},
    {"name": "GitPull", "cmd": ["git", "pull", "origin", "main"], "interval": 6},
]

last_run = {t["name"]: None for t in TASKS}
stats = {"cycles": 0, "success": 0, "fail": 0}


async def run_cmd(cmd: list, timeout: int = 600):
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            cwd=PROJECT_DIR, env={**os.environ, "PYTHONPATH": str(PROJECT_DIR)}
        )
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return proc.returncode == 0, out.decode("utf-8", errors="replace")
    except asyncio.TimeoutError:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)


async def run_task(task: dict):
    name = task["name"]
    interval = task["interval"]
    last = last_run.get(name)
    if last and (datetime.now() - last).total_seconds() < interval * 3600:
        return None

    logger.info(f"[{name}] Starting...")
    start = time.time()
    ok, out = await run_cmd(task["cmd"], timeout=interval * 900)
    elapsed = time.time() - start
    last_run[name] = datetime.now()

    if ok:
        stats["success"] += 1
        logger.info(f"[{name}] OK ({elapsed:.1f}s)")
    else:
        stats["fail"] += 1
        logger.warning(f"[{name}] FAIL: {out[:150]}")

    return ok


async def cycle():
    logger.info("=" * 40)
    logger.info(f"CYCLE {stats['cycles'] + 1}")
    logger.info("=" * 40)
    for task in TASKS:
        await run_task(task)
        await asyncio.sleep(2)
    stats["cycles"] += 1
    logger.info(f"Stats: s={stats['success']}, f={stats['fail']}")


async def main():
    logger.info("Godspeed Auto-Evolver 24/7 STARTING")
    logger.info(f"Project: {PROJECT_DIR}")
    logger.info(f"Cycle: every {CYCLE_INTERVAL_HOURS}h")

    while True:
        try:
            await cycle()
        except Exception as e:
            logger.error(f"Cycle error: {e}")
            stats["fail"] += 1
        logger.info(f"Sleeping {CYCLE_INTERVAL_HOURS}h...")
        await asyncio.sleep(CYCLE_INTERVAL_HOURS * 3600)


if __name__ == "__main__":
    asyncio.run(main())