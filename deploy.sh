#!/bin/bash
# Deploy script for Orbit agent to Databricks Apps

set -e

echo "=== Deploying Orbit Agent to Databricks Apps ==="

# Get current user
echo "Getting Databricks username..."
DATABRICKS_USERNAME=$(databricks current-user me | jq -r .userName)
echo "Username: $DATABRICKS_USERNAME"

# Sync files
echo ""
echo "Syncing files to workspace..."
cd agent-langgraph
databricks sync . "/Users/$DATABRICKS_USERNAME/agent-langgraph"

# Deploy app
echo ""
echo "Deploying app..."
databricks apps deploy agent-langgraph --source-code-path "/Workspace/Users/$DATABRICKS_USERNAME/agent-langgraph"

echo ""
echo "=== Deployment complete! ==="
echo "The agent routing loop fix has been deployed."
echo "Check your Databricks Apps dashboard to monitor the deployment status."
