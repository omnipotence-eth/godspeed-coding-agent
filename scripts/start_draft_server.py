"""Start llama-server with 1.5B GPU draft (~750 tok/s)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from godspeed.tools.llamacpp_manager import DEFAULT_URL, is_server_running, start_server

proc = start_server()
if proc:
    print("Server started, PID:", proc.pid, flush=True)
    for _ in range(20):
        time.sleep(1)
        if is_server_running(DEFAULT_URL):
            print("Server READY at", DEFAULT_URL, flush=True)
            break
    else:
        print("Server did not become ready", flush=True)
        if proc.poll() is not None:
            out, err = proc.communicate(timeout=5)
            print("STDOUT:", out[-500:])
            print("STDERR:", err[-500:])
elif is_server_running(DEFAULT_URL):
    print("Server was already running at", DEFAULT_URL, flush=True)
else:
    print("Failed to start server", flush=True)
    sys.exit(1)
