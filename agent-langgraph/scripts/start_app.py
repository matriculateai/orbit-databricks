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

                    if self.frontend_ready:
                        print("\n" + "=" * 50)
                        print("✓ Orbit frontend is ready!")
                        print("✓ Using deployed Databricks backend endpoint")
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
        print("Shutting down frontend process...")
        print("=" * 42)

        if self.frontend_process:
            try:
                self.frontend_process.terminate()
                self.frontend_process.wait(timeout=5)
            except (subprocess.TimeoutExpired, Exception):
                self.frontend_process.kill()

        if self.frontend_log:
            self.frontend_log.close()

    def run(self):
        load_dotenv(dotenv_path=".env.local", override=True)

        # Open log files
        self.frontend_log = open("frontend.log", "w", buffering=1)

        try:
            # Use deployed Databricks endpoint instead of local backend
            backend_url = os.environ.get(
                "BACKEND_URL",
                "https://dbc-2dd00323-bb3d.cloud.databricks.com/serving-endpoints/orbit-multiagent/invocations"
            )

            print(f"Using backend endpoint: {backend_url}")

            # Mark backend as ready since we're not starting it locally
            self.backend_ready = True

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
                    "BACKEND_URL": backend_url,
                    "DATABRICKS_TOKEN": os.environ.get("DATABRICKS_TOKEN", ""),
                    # Databricks configuration for dashboard embedding
                    "DATABRICKS_HOST": os.environ.get("DATABRICKS_HOST", ""),
                    "DATABRICKS_WORKSPACE_ID": os.environ.get("DATABRICKS_WORKSPACE_ID", ""),
                    # Dashboard IDs for AIBI client
                    "ORBIT_REP_MANAGER_DASHBOARD_ID": os.environ.get("ORBIT_REP_MANAGER_DASHBOARD_ID", ""),
                    "ORBIT_EXEC_DASHBOARD_ID": os.environ.get("ORBIT_EXEC_DASHBOARD_ID", ""),
                },
            )

            print(
                f"\nMonitoring frontend process (PID: {self.frontend_process.pid})\n"
            )

            # Wait for frontend failure
            while not self.failed.is_set():
                time.sleep(0.1)
                if self.frontend_process.poll() is not None:
                    self.failed.set()
                    break

            # Frontend failed
            exit_code = self.frontend_process.returncode if self.frontend_process else 1

            print(
                f"\n{'=' * 42}\nERROR: frontend process exited with code {exit_code}\n{'=' * 42}"
            )
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
