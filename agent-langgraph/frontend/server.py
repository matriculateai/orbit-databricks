#!/usr/bin/env python3
"""
Orbit Frontend Server
Serves static files and proxies API requests to the backend agent.
"""

import json
import os
import sys
import traceback
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

# Configuration
FRONTEND_PORT = int(os.environ.get("FRONTEND_PORT", "8000"))
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8001/invocations")

# Databricks configuration
DATABRICKS_HOST = os.environ.get("DATABRICKS_HOST", "")
WORKSPACE_ID = os.environ.get("DATABRICKS_WORKSPACE_ID", "")

# Dashboard URLs (Configured in .env)
# Expected format: https://<host>/dashboards/<id>/published?embedded=true
REP_MANAGER_DASHBOARD_URL = os.environ.get("ORBIT_REP_MANAGER_DASHBOARD_URL", "")
EXEC_DASHBOARD_URL = os.environ.get("ORBIT_EXEC_DASHBOARD_URL", "")


class OrbitHandler(SimpleHTTPRequestHandler):
    """Custom HTTP handler for Orbit frontend."""

    def __init__(self, *args, **kwargs):
        self.directory = str(Path(__file__).parent)
        super().__init__(*args, directory=self.directory, **kwargs)

    def do_GET(self):
        """Handle GET requests."""
        if self.path == "/api/config":
            self.send_config()
        elif self.path == "/" or self.path == "/index.html":
            self.path = "/index.html"
            super().do_GET()
        else:
            super().do_GET()

    def do_POST(self):
        """Handle POST requests."""
        if self.path == "/api/chat":
            self.handle_chat()
        else:
            self.send_error(404, "Not Found")

    def send_config(self):
        """Send configuration to frontend."""
        config = {
            "databricksHost": DATABRICKS_HOST,
            "workspaceId": WORKSPACE_ID,
            # Sending full URLs instead of IDs
            "repManagerDashboardUrl": REP_MANAGER_DASHBOARD_URL,
            "execDashboardUrl": EXEC_DASHBOARD_URL,
        }
        self.send_json_response(config)

    def handle_chat(self):
        """Proxy chat requests to backend agent."""
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            data = json.loads(body)

            message = data.get("message", "")
            context = data.get("context")

            print(f"[Frontend] Received message: {message[:50]}...")

            backend_payload = {
                "input": [{"role": "user", "content": message}],
            }

            if context:
                backend_payload["custom_inputs"] = {"context": context}

            print(f"[Frontend] Sending to backend: {BACKEND_URL}")

            req = Request(
                BACKEND_URL,
                data=json.dumps(backend_payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            with urlopen(req, timeout=300) as response:
                result = json.loads(response.read().decode("utf-8"))
                print(f"[Frontend] Backend response keys: {result.keys()}")

            # Simple extraction logic
            response_text = ""
            if "output" in result and result["output"]:
                # Handle MLflow output format
                for item in result["output"]:
                    if isinstance(item, dict) and item.get("type") == "text":
                        content = item.get("content", "")
                        # Skip internal node markers
                        if not content.startswith("<n>") and not content.startswith("__ORBIT_CONTEXT_UPDATE__"):
                            response_text += content

            if not response_text and "message" in result:
                response_text = result["message"]

            # Fallback if no response text found
            if not response_text:
                print(f"[Warning] No response text found in backend result: {result}")
                response_text = "I apologize, but I couldn't generate a response. Please try again."

            new_context = None
            if "custom_outputs" in result:
                new_context = result["custom_outputs"].get("context")

            self.send_json_response({
                "response": response_text,
                "context": new_context,
            })

        except Exception as e:
            print(f"[Error] Exception in handle_chat: {e}")
            self.send_json_response(
                {"error": str(e), "response": "An unexpected error occurred."},
                status=500,
            )

    def send_json_response(self, data, status=200):
        response_body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(response_body))
        self.end_headers()
        self.wfile.write(response_body)


def main():
    server_address = ("", FRONTEND_PORT)
    httpd = HTTPServer(server_address, OrbitHandler)
    print(f"Server is running on http://localhost:{FRONTEND_PORT}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.shutdown()


if __name__ == "__main__":
    main()
