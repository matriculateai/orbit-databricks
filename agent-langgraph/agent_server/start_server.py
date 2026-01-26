import os

import uvicorn
from dotenv import load_dotenv
from mlflow.genai.agent_server import AgentServer, setup_mlflow_git_based_version_tracking

# Load env vars from .env.local before importing the agent for proper auth
load_dotenv(dotenv_path=".env.local", override=True)

# Need to import the agent to register the functions with the server
import agent_server.agent  # noqa: E402

# Disable chat proxy since frontend handles it
agent_server = AgentServer("ResponsesAgent", enable_chat_proxy=False)

# Define the app as a module level variable to enable multiple workers
app = agent_server.app  # noqa: F841
setup_mlflow_git_based_version_tracking()


def main():
    # Backend runs on port 8001 (frontend on 8000 handles external traffic)
    port = int(os.environ.get("BACKEND_PORT", "8001"))
    uvicorn.run("agent_server.start_server:app", host="0.0.0.0", port=port)
