#!/usr/bin/env python3
"""
Start script for running frontend and backend processes concurrently.

Requirements:
1. Not reporting ready until BOTH frontend and backend processes are ready
2. Exiting as soon as EITHER process fails
3. Printing error logs if either process fails
"""

import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

from dotenv import load_dotenv

# Port configuration
# Frontend runs on 8000 (Databricks Apps exposed port)
# Backend runs on 8001 (internal only)
FRONTEND_PORT = 8000
BACKEND_PORT = 8001

# Readiness patterns
BACKEND_READY = [r"Uvicorn running on", r"Application startup complete", r"Started server process"]
FRONTEND_READY = [r"Server is running on http://localhost"]


class ProcessManager:
    def __init__(self):
        self.backend_process = None
        self.frontend_process = None
        self.backend_ready = False
        self.frontend_ready = False
        self.failed = threading.Event()
        self.backend_log = None
        self.frontend_log = None

    def monitor_process(self, process, name, log_file, patterns):
        is_ready = False
        try:
            for line in iter(process.stdout.readline, ""):
                if not line:
                    break

                line = line.rstrip()
                log_file.write(line + "\n")
                print(f"[{name}] {line}")

                # Check readiness
                if not is_ready and any(re.search(p, line, re.IGNORECASE) for p in patterns):
                    is_ready = True
                    if name == "backend":
                        self.backend_ready = True
                    else:
                        self.frontend_ready = True
                    print(f"✓ {name.capitalize()} is ready!")

                    if self.backend_ready and self.frontend_ready:
                        print("\n" + "=" * 50)
                        print("✓ Both frontend and backend are ready!")
                        print("✓ Orbit is running at http://localhost:8000")
                        print("=" * 50 + "\n")

            process.wait()
            if process.returncode != 0:
                self.failed.set()

        except Exception as e:
            print(f"Error monitoring {name}: {e}")
            self.failed.set()

    def start_process(self, cmd, name, log_file, patterns, cwd=None, env_extra=None):
        print(f"Starting {name}...")
        env = os.environ.copy()
        if env_extra:
            env.update(env_extra)
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, cwd=cwd, env=env
        )

        thread = threading.Thread(
            target=self.monitor_process, args=(process, name, log_file, patterns), daemon=True
        )
        thread.start()
        return process

    def print_logs(self, log_path):
        print(f"\nLast 50 lines of {log_path}:")
        print("-" * 40)
        try:
            lines = Path(log_path).read_text().splitlines()
            print("\n".join(lines[-50:]))
        except FileNotFoundError:
            print(f"(no {log_path} found)")
        print("-" * 40)

    def cleanup(self):
        print("\n" + "=" * 42)
        print("Shutting down both processes...")
        print("=" * 42)

        for proc in [self.backend_process, self.frontend_process]:
            if proc:
                try:
                    proc.terminate()
                    proc.wait(timeout=5)
                except (subprocess.TimeoutExpired, Exception):
                    proc.kill()

        if self.backend_log:
            self.backend_log.close()
        if self.frontend_log:
            self.frontend_log.close()

    def run(self):
        load_dotenv(dotenv_path=".env.local", override=True)

        # Open log files
        self.backend_log = open("backend.log", "w", buffering=1)
        self.frontend_log = open("frontend.log", "w", buffering=1)

        try:
            # Start backend on port 8001 (internal)
            self.backend_process = self.start_process(
                ["uv", "run", "start-server"],
                "backend",
                self.backend_log,
                BACKEND_READY,
                env_extra={"BACKEND_PORT": str(BACKEND_PORT)},
            )

            # Start frontend on port 8000 (exposed by Databricks Apps)
            frontend_dir = Path(__file__).parent.parent / "frontend"
            self.frontend_process = self.start_process(
                ["python", "server.py"],
                "frontend",
                self.frontend_log,
                FRONTEND_READY,
                cwd=frontend_dir,
                env_extra={
                    "FRONTEND_PORT": str(FRONTEND_PORT),
                    "BACKEND_URL": f"http://localhost:{BACKEND_PORT}/invocations",
                    # Pass through dashboard URLs from app.yaml
                    "ORBIT_REP_MANAGER_DASHBOARD_URL": os.environ.get("ORBIT_REP_MANAGER_DASHBOARD_URL", ""),
                    "ORBIT_EXEC_DASHBOARD_URL": os.environ.get("ORBIT_EXEC_DASHBOARD_URL", ""),
                },
            )

            print(
                f"\nMonitoring processes (Backend PID: {self.backend_process.pid}, Frontend PID: {self.frontend_process.pid})\n"
            )

            # Wait for failure
            while not self.failed.is_set():
                time.sleep(0.1)
                for proc in [self.backend_process, self.frontend_process]:
                    if proc.poll() is not None:
                        self.failed.set()
                        break

            # Determine which failed
            failed_name = "backend" if self.backend_process.poll() is not None else "frontend"
            failed_proc = (
                self.backend_process if failed_name == "backend" else self.frontend_process
            )
            exit_code = failed_proc.returncode if failed_proc else 1

            print(
                f"\n{'=' * 42}\nERROR: {failed_name} process exited with code {exit_code}\n{'=' * 42}"
            )
            self.print_logs("backend.log")
            self.print_logs("frontend.log")
            return exit_code

        except KeyboardInterrupt:
            print("\nInterrupted")
            return 0

        finally:
            self.cleanup()


def main():
    sys.exit(ProcessManager().run())


if __name__ == "__main__":
    main()
