"""Serve the API and built React dashboard on the Space's single HTTP port."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from threading import Event


def main():
    stopping = Event()

    def stop(_signum, _frame):
        stopping.set()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    build_file = Path(__file__).with_name("build.json")
    if build_file.is_file():
        os.environ["SLEEPOPS_BUILD_ID"] = json.loads(build_file.read_text(encoding="utf-8"))["build_id"]
    nginx = Path(__file__).with_name("nginx.conf").resolve()
    services = (
        (sys.executable, "-m", "uvicorn", "sleepops.api:app", "--host", "127.0.0.1", "--port", "8000",
         "--root-path", "/api"),
        ("nginx", "-c", str(nginx), "-g", "daemon off;"),
    )
    processes = []
    try:
        for command in services:
            processes.append(subprocess.Popen(command))
        while not stopping.wait(0.5):
            for process in processes:
                status = process.poll()
                if status is not None:
                    raise RuntimeError(f"{process.args[0]} exited with status {status}.")
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    main()
