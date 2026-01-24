#!/usr/bin/env python3
"""
Orbit Frontend Server
Serves static files and proxies API requests to the backend agent.
"""

import json
import os
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


# Configuration
FRONTEND_PORT = int(os.environ.get("CHAT_APP_PORT", "3000"))
BACKEND_URL = os.environ.get("API_PROXY", "http://localhost:8000/invocations")
REP_MANAGER_DASHBOARD_URL = os.environ.get("ORBIT_REP_MANAGER_DASHBOARD_URL", "")
EXEC_DASHBOARD_URL = os.environ.get("ORBIT_EXEC_DASHBOARD_URL", "")


class OrbitHandler(SimpleHTTPRequestHandler):
    """Custom HTTP handler for Orbit frontend."""

    def __init__(self, *args, **kwargs):
        # Set the directory to serve files from
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
            "repManagerDashboardUrl": REP_MANAGER_DASHBOARD_URL,
            "execDashboardUrl": EXEC_DASHBOARD_URL,
        }
        self.send_json_response(config)

    def handle_chat(self):
        """Proxy chat requests to backend agent."""
        try:
            # Read request body
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            data = json.loads(body)

            # Extract message and context
            message = data.get("message", "")
            context = data.get("context")

            # Build request for backend (MLflow Responses API format)
            backend_payload = {
                "input": [{"role": "user", "content": message}],
            }

            if context:
                backend_payload["custom_inputs"] = {"context": context}

            # Send to backend
            req = Request(
                BACKEND_URL,
                data=json.dumps(backend_payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            with urlopen(req, timeout=300) as response:
                result = json.loads(response.read().decode("utf-8"))

            # Extract response text from MLflow format
            response_text = ""
            if "output" in result:
                for item in result["output"]:
                    if item.get("type") == "text":
                        response_text = item.get("content", "")
                        break
            elif "content" in result:
                response_text = result["content"]
            else:
                response_text = str(result)

            # Extract updated context
            new_context = None
            if "custom_outputs" in result:
                new_context = result["custom_outputs"].get("context")

            # Send response
            self.send_json_response({
                "response": response_text,
                "context": new_context,
            })

        except HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else str(e)
            print(f"Backend error: {e.code} - {error_body}")
            self.send_json_response(
                {"error": f"Backend error: {e.code}", "response": "I encountered an error processing your request."},
                status=500,
            )
        except URLError as e:
            print(f"Connection error: {e}")
            self.send_json_response(
                {"error": "Backend unavailable", "response": "The backend service is currently unavailable."},
                status=503,
            )
        except Exception as e:
            print(f"Error handling chat: {e}")
            self.send_json_response(
                {"error": str(e), "response": "An unexpected error occurred."},
                status=500,
            )

    def send_json_response(self, data, status=200):
        """Send JSON response."""
        response_body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(response_body))
        self.end_headers()
        self.wfile.write(response_body)

    def log_message(self, format, *args):
        """Custom logging format."""
        print(f"[Frontend] {args[0]}")


def main():
    """Start the frontend server."""
    server_address = ("", FRONTEND_PORT)
    httpd = HTTPServer(server_address, OrbitHandler)
    print(f"Server is running on http://localhost:{FRONTEND_PORT}")
    print(f"Backend API: {BACKEND_URL}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.shutdown()


if __name__ == "__main__":
    main()
