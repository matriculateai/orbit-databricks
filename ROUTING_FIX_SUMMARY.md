# Agent Routing Loop Fix - Deployment Instructions

## The Problem

The chat agent was stuck in a routing loop:
- Turn 1: Routes to sales_agent ✓
- Turn 2-13+: Keeps routing to fallback_agent ✗ (should go to reasoning → END)

**Root cause:** GenieAgent messages had `name='query_result'` instead of the agent name (e.g., `name='sales_agent'`). The supervisor couldn't recognize which agent sent the message, so it treated every response as new user input.

## The Solution

Implemented **tagger nodes** that intercept agent messages and set the correct `name` attribute:

### Architecture Changes
```
Before:
User → Supervisor → sales_agent → Supervisor (doesn't recognize) → Routes again → LOOP

After:
User → Supervisor → sales_agent → sales_agent_tagger → Supervisor (recognizes!) → reasoning_agent → reasoning_agent_tagger → Supervisor → END
```

### Code Changes (agent.py)
1. **Lines 182-229**: Created `create_message_tagger()` function
   - Detects AIMessage objects by checking `type().__name__ == 'AIMessage'`
   - Converts to dict with explicit `name` field set to agent name
   - Adds debug logging to track tagging

2. **Lines 399-419**: Updated graph structure
   - Added tagger nodes for each agent
   - Set up edges: agent → tagger → supervisor
   - Ensures all assistant messages have proper name attribution

3. **Lines 151-180**: Simplified agent initialization
   - Removed broken wrapper approach
   - Use GenieAgent and create_react_agent directly
   - Let taggers handle name attribution post-processing

## What the Fix Does

### Message Flow Example
```
1. User asks: "Show me sales for Pantoprazole"
2. Supervisor routes to sales_agent (Turn 1)
3. sales_agent creates: AIMessage(name='query_result', content='...')
4. sales_agent_tagger converts to: {'role': 'assistant', 'name': 'sales_agent', 'content': '...'}
5. Supervisor sees msg_name='sales_agent' → routes to reasoning_agent
6. reasoning_agent creates message
7. reasoning_agent_tagger tags with name='reasoning_agent'
8. Supervisor sees msg_name='reasoning_agent' → routes to END ✓
```

### Expected Debug Logs
```
Turn 1 | Route: sales_agent | Intent: sales_query
DEBUG: Tagger sales_agent processing message type: AIMessage
DEBUG: Tagged message from sales_agent, original name was: query_result
DEBUG: last_message type: <class 'dict'>, msg_name: sales_agent, role: assistant
```

## Deployment Instructions

### Quick Deploy
Run the deployment script:
```bash
cd /home/user/orbit-databricks
./deploy.sh
```

### Manual Deploy
If the script fails, deploy manually:

```bash
# Get your username
DATABRICKS_USERNAME=$(databricks current-user me | jq -r .userName)

# Sync files to workspace
cd agent-langgraph
databricks sync . "/Users/$DATABRICKS_USERNAME/agent-langgraph"

# Deploy the app
databricks apps deploy agent-langgraph --source-code-path "/Workspace/Users/$DATABRICKS_USERNAME/agent-langgraph"
```

## Verification

After deployment, check the logs for:
1. ✓ "DEBUG: Tagger {agent_name} processing message type: AIMessage"
2. ✓ "DEBUG: Tagged message from {agent_name}"
3. ✓ Turn counter stays at 1 (doesn't increment to 2, 3, 4...)
4. ✓ Supervisor recognizes agent names and routes to reasoning → END

## Troubleshooting

### If the loop still occurs:
1. **Check deployment status**: `databricks apps get agent-langgraph`
2. **View live logs**: Check Databricks Apps dashboard
3. **Verify code is deployed**: Check if tagger debug logs appear
4. **Check agent names**: Ensure genie_agent_names list matches tagger node names

### About the /serving-endpoints/chat/completions endpoint:
- This is the Databricks LLM serving endpoint (returns 200 OK ✓)
- Called by `structured_llm.invoke()` at agent.py:310
- **Should only be called once per user query** (for routing decision)
- If called multiple times = routing loop still present

## Files Changed
- `agent-langgraph/agent_server/agent.py`: All routing loop fixes
- `deploy.sh`: Deployment script (new)
- `ROUTING_FIX_SUMMARY.md`: This document (new)

## Branch
- **Branch**: `claude/fix-agent-routing-loop-cIClw`
- **Latest commit**: 6d4a24b
- **Status**: Ready for deployment

## Next Steps
1. Run `./deploy.sh` to deploy the fix
2. Test with a query like "Show me sales for Pantoprazole"
3. Verify logs show tagger debug messages
4. Confirm no routing loop (turn counter stays at 1)
