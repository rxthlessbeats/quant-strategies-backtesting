"""Start API + Next.js with one `uv run start`."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
FRONTEND = ROOT / "frontend"


def _stop(procs: list[subprocess.Popen[bytes]]) -> None:
    for proc in procs:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
    deadline = time.time() + 8
    for proc in procs:
        remaining = max(0.1, deadline - time.time())
        try:
            proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            proc.kill()


def main() -> None:
    npm = shutil.which("npm")
    if npm is None:
        sys.exit("npm not found. Node 20 is pinned in .mise.toml (mise install).")

    if not (FRONTEND / "node_modules").exists():
        subprocess.run(
            [npm, "ci", "--legacy-peer-deps"],
            cwd=FRONTEND,
            check=True,
        )

    env = os.environ.copy()
    procs: list[subprocess.Popen[bytes]] = [
        subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "app.main:app",
                "--reload",
                "--host",
                "127.0.0.1",
                "--port",
                "8000",
            ],
            cwd=BACKEND,
            env=env,
        ),
        subprocess.Popen(
            [npm, "run", "dev"],
            cwd=FRONTEND,
            env=env,
        ),
    ]

    def handle_signal(_signum: int, _frame: object | None) -> None:
        _stop(procs)
        raise SystemExit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    print("API  http://127.0.0.1:8000")
    print("App  http://localhost:3000")
    print("Stop with Ctrl+C")

    try:
        while True:
            for proc in procs:
                code = proc.poll()
                if code is not None:
                    _stop(procs)
                    raise SystemExit(code)
            time.sleep(0.4)
    except KeyboardInterrupt:
        handle_signal(signal.SIGINT, None)


if __name__ == "__main__":
    main()
