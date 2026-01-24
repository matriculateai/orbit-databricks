"""
Orbit Multi-Agent System - Databricks App Deployment
Production-ready pharmaceutical intelligence agent with multi-turn conversation support.
"""

from typing import AsyncGenerator, Optional, Literal, TypedDict
from uuid import uuid4
from datetime import datetime
import json
import logging
import os

import mlflow
from databricks.sdk import WorkspaceClient
from databricks_langchain import ChatDatabricks, GenieAgent
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from mlflow.genai.agent_server import invoke, stream
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
    output_to_responses_items_stream,
    to_chat_completions_input,
)
from pydantic import BaseModel, Field

from agent_server.utils import (
    get_databricks_host_from_env,
    get_user_workspace_client,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mlflow.langchain.autolog()
sp_workspace_client = WorkspaceClient()

# ==================== CONFIGURATION FROM ENV ====================

SUPERVISOR = "supervisor"
REASONING = "reasoning_agent"

# Genie Space IDs from environment variables
GENIE_SPACES = {
    "sales": os.getenv("ORBIT_SALES_SPACE_ID", "01f0f67aa809b0185446d9e3"),
    "stock": os.getenv("ORBIT_STOCK_SPACE_ID", "01f0f690ce7d36f0bd646318"),
    "reps": os.getenv("ORBIT_REPS_SPACE_ID", "01f0f692052188f1ab70963948"),
    "fallback": os.getenv("ORBIT_FALLBACK_SPACE_ID", "01f0f692a4a070316fe4cd3323"),
}

# LLM Endpoint from environment
LLM_ENDPOINT_NAME = os.getenv("ORBIT_LLM_ENDPOINT", "databricks-gemma-3-12b")

# Log configuration on startup
logger.info(f"Orbit Agent Configuration:")
logger.info(f"  LLM Endpoint: {LLM_ENDPOINT_NAME}")
logger.info(f"  Genie Spaces: {list(GENIE_SPACES.keys())}")
logger.info(f"  MLflow Tracking URI: {os.getenv('MLFLOW_TRACKING_URI', 'not set')}")
logger.info(f"  MLflow Registry URI: {os.getenv('MLFLOW_REGISTRY_URI', 'not set')}")


# ==================== STATE MODELS ====================

class ConversationContext(TypedDict, total=False):
    """Multi-turn conversation state."""
    conversation_id: str
    turn_count: int
    last_query_time: str
    last_intent: Optional[str]
    last_agent_used: Optional[str]
    waiting_for_clarification: bool
    clarification_context: Optional[str]
    current_product: Optional[str]
    current_customer: Optional[str]
    current_region: Optional[str]
    current_rep: Optional[str]
    current_time_period: Optional[str]
    query_history: list
    refinement_suggestions: list


class OrbitState(MessagesState):
    """Extended state with conversation context."""
    context: ConversationContext


class ExtractedEntities(BaseModel):
    """Entities extracted from user query."""
    product: Optional[str] = Field(default=None, description="Product name or code")
    customer: Optional[str] = Field(default=None, description="Customer/pharmacy name")
    region: Optional[str] = Field(default=None, description="Geographic region")
    rep: Optional[str] = Field(default=None, description="Sales rep name or code")
    time_period: Optional[str] = Field(default=None, description="Time period (e.g., 'last month')")


class RoutingDecision(BaseModel):
    """Comprehensive supervisor decision (single LLM call instead of 3)."""
    entities: ExtractedEntities = Field(description="Extracted entities from query")
    is_ambiguous: bool = Field(description="Is query too vague to answer?")
    missing_info: Optional[list[str]] = Field(default=None, description="Missing information if ambiguous")
    suggested_questions: Optional[list[str]] = Field(default=None, description="Example clarifying questions")
    intent_type: Literal["greeting", "off_topic", "sales_query", "stock_query", "rep_query", "complex_query"] = Field(
        description="Primary intent of query"
    )
    next_action: Literal["sales_agent", "stock_agent", "reps_agent", "fallback_agent", "direct_response"] = Field(
        description="Which agent to route to"
    )
    response_content: Optional[str] = Field(default=None, description="Direct response for greetings/off-topic")
    reasoning: str = Field(description="Routing decision explanation")


# ==================== HELPER FUNCTIONS ====================

def get_msg_attr(msg, attr, default=None):
    """Get message attribute, handling both dict and object types."""
    if hasattr(msg, attr):
        return getattr(msg, attr)
    elif isinstance(msg, dict):
        return msg.get(attr, default)
    return default


# ==================== SUPERVISOR CREATION ====================

def create_orbit_supervisor(workspace_client: Optional[WorkspaceClient] = None):
    """
    Create the Orbit multi-agent supervisor.
    
    Architecture:
    - Genie agents for data queries (sales, stock, reps)
    - Reasoning agent for synthesis
    - Supervisor for routing with structured outputs
    """
    
    llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)
    
    # Initialize Genie agents
    genie_agents = {
        "sales_agent": GenieAgent(
            genie_space_id=GENIE_SPACES["sales"],
            genie_agent_name="sales_agent",
            description="Answers questions about sales performance, revenue trends, and targets.",
        ),
        "stock_agent": GenieAgent(
            genie_space_id=GENIE_SPACES["stock"],
            genie_agent_name="stock_agent",
            description="Answers questions about inventory levels, stockout risks, and backorders.",
        ),
        "reps_agent": GenieAgent(
            genie_space_id=GENIE_SPACES["reps"],
            genie_agent_name="reps_agent",
            description="Answers questions about sales rep activity and performance.",
        ),
        "fallback_agent": GenieAgent(
            genie_space_id=GENIE_SPACES["fallback"],
            genie_agent_name="fallback_agent",
            description="Handles complex queries spanning multiple domains.",
        ),
    }
    
    # Initialize reasoning agent
    reasoning_agent = create_react_agent(
        model=ChatDatabricks(endpoint=LLM_ENDPOINT_NAME),
        tools=[],
        name=REASONING,
    )
    
    # Agent descriptions for prompting
    agent_descriptions = "\n".join([
        f"- **{name}**: {agent.description}"
        for name, agent in genie_agents.items()
    ])
    
    # Define supervisor routing logic
    def supervisor_node(state: OrbitState) -> Command[Literal[
        "sales_agent", "stock_agent", "reps_agent", "fallback_agent", "reasoning_agent", "__end__"
    ]]:
        """Optimized supervisor using structured outputs."""
        
        messages = state["messages"]
        context = state.get("context", {})
        last_message = messages[-1]
        
        # Initialize context if first turn
        if not context:
            context = {
                "conversation_id": str(uuid4()),
                "turn_count": 0,
                "query_history": [],
                "waiting_for_clarification": False,
            }
        
        # Handle returns from workers (don't increment turn)
        msg_name = get_msg_attr(last_message, 'name')
        genie_agent_names = ["sales_agent", "stock_agent", "reps_agent", "fallback_agent"]
        
        if msg_name in genie_agent_names:
            context["last_agent_used"] = msg_name
            return Command(
                goto=REASONING,
                update={"context": context}
            )
        
        if msg_name == REASONING:
            return Command(goto=END, update={"context": context})
        
        # Process new user input (increment turn only here)
        context["turn_count"] = context.get("turn_count", 0) + 1
        context["last_query_time"] = datetime.now().isoformat()
        
        user_query = get_msg_attr(last_message, 'content', str(last_message))
        
        # Build context-aware prompt
        system_prompt = f"""You are the Orbit Supervisor, routing pharmaceutical intelligence queries.

## AVAILABLE AGENTS
{agent_descriptions}

## CONVERSATION CONTEXT
Turn: {context['turn_count']}
Product: {context.get('current_product', 'Not set')}
Region: {context.get('current_region', 'Not set')}
Time Period: {context.get('current_time_period', 'Not set')}

## ROUTING RULES (PRIORITY ORDER)
1. **greetings** (hi, hello, thanks) → next_action: "direct_response"
2. **off_topic** (coding, recipes, etc.) → next_action: "direct_response"
3. **sales_query** → next_action: "sales_agent"
4. **stock_query** → next_action: "stock_agent"
5. **rep_query** → next_action: "reps_agent"
6. **complex_query** → next_action: "fallback_agent"

CRITICAL: Never route greetings to clarification. Greetings get direct_response.

## CURRENT QUERY
"{user_query}"

Provide your decision using the RoutingDecision schema."""
        
        # Single structured output call (replaces 3 sequential calls)
        structured_llm = llm.with_structured_output(RoutingDecision)
        
        try:
            decision: RoutingDecision = structured_llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ])
            
            # Validate and log
            if decision.next_action not in ["sales_agent", "stock_agent", "reps_agent", "fallback_agent", "direct_response"]:
                logger.warning(f"Invalid action '{decision.next_action}', using fallback")
                decision.next_action = "fallback_agent"
            
            logger.info(f"Turn {context['turn_count']} | Route: {decision.next_action} | Intent: {decision.intent_type}")
            
        except Exception as e:
            logger.error(f"Structured output failed: {e}", exc_info=True)
            decision = RoutingDecision(
                entities=ExtractedEntities(),
                is_ambiguous=False,
                intent_type="complex_query",
                next_action="fallback_agent",
                reasoning=f"Error fallback: {str(e)}"
            )
        
        # Update context with extracted entities
        if decision.entities.product:
            context["current_product"] = decision.entities.product
        if decision.entities.region:
            context["current_region"] = decision.entities.region
        if decision.entities.time_period:
            context["current_time_period"] = decision.entities.time_period
        
        # Update query history
        query_history = context.get("query_history", [])
        query_history.append({
            "turn": context["turn_count"],
            "query": user_query,
            "entities": {
                "product": context.get("current_product"),
                "region": context.get("current_region"),
                "time_period": context.get("current_time_period"),
            },
            "intent": decision.intent_type,
        })
        context["query_history"] = query_history[-5:]
        context["last_intent"] = decision.intent_type
        
        # Handle greetings and off-topic
        if decision.intent_type in ["greeting", "off_topic"] or decision.next_action == "direct_response":
            response_text = decision.response_content or (
                "Hi! I'm Orbit, your pharmaceutical intelligence assistant. "
                "I can help you analyze sales data, stock levels, rep performance, and targets. "
                "What would you like to know?"
            )
            return Command(
                goto=END,
                update={
                    "context": context,
                    "messages": [{"role": "assistant", "content": response_text}]
                }
            )
        
        # Handle ambiguity (only for data queries)
        if decision.is_ambiguous and decision.intent_type not in ["greeting", "off_topic"]:
            missing_str = ", ".join(decision.missing_info or ["specific details"])
            suggestions = decision.suggested_questions or [
                "Sales for Pantoprazole in December 2025",
                "Stock levels for Pain category in Western Cape"
            ]
            suggestions_str = "\n".join([f"- {q}" for q in suggestions])
            
            clarification = f"""I'd be happy to help! To give you the most relevant information, could you let me know {missing_str}?

For example:
{suggestions_str}"""
            
            return Command(
                goto=END,
                update={
                    "context": context,
                    "messages": [{"role": "assistant", "content": clarification}]
                }
            )
        
        # Route to Genie agent
        target_agent = decision.next_action
        if target_agent not in genie_agent_names:
            target_agent = "fallback_agent"
        
        return Command(goto=target_agent, update={"context": context})
    
    # Build graph
    workflow = StateGraph(OrbitState)
    workflow.add_node(SUPERVISOR, supervisor_node)
    
    for name, agent in genie_agents.items():
        workflow.add_node(name, agent)
    
    workflow.add_node(REASONING, reasoning_agent)
    
    workflow.add_edge(START, SUPERVISOR)
    for name in genie_agents.keys():
        workflow.add_edge(name, SUPERVISOR)
    workflow.add_edge(REASONING, SUPERVISOR)
    
    return workflow.compile()


# ==================== AGENT INITIALIZATION ====================

_orbit_agent = None


async def init_agent(workspace_client: Optional[WorkspaceClient] = None):
    """Initialize or return cached Orbit supervisor agent."""
    global _orbit_agent
    if _orbit_agent is None:
        _orbit_agent = create_orbit_supervisor(workspace_client or sp_workspace_client)
    return _orbit_agent


# ==================== API ENDPOINTS ====================

@invoke()
async def non_streaming(request: ResponsesAgentRequest) -> ResponsesAgentResponse:
    """Synchronous endpoint with proper context handling."""
    agent = await init_agent()
    cc_msgs = to_chat_completions_input([i.model_dump() for i in request.input])
    
    # Extract initial context from request
    initial_context = request.custom_inputs.get("context", {}) if request.custom_inputs else {}
    
    # Run agent to completion
    final_state = agent.invoke({"messages": cc_msgs, "context": initial_context})
    
    # Extract updated context
    final_context = final_state.get("context", {})
    
    # Extract assistant messages
    outputs = []
    messages = final_state.get("messages", [])
    
    # Ensure messages is a list
    if not isinstance(messages, list):
        messages = [messages]
    
    for msg in messages:
        msg_role = get_msg_attr(msg, 'role')
        msg_name = get_msg_attr(msg, 'name')
        
        if msg_role == 'assistant' and msg_name not in ['sales_agent', 'stock_agent', 'reps_agent', 'fallback_agent', REASONING]:
            content = get_msg_attr(msg, 'content', '')
            if content and not content.startswith('<n>'):
                from mlflow.types.responses import ResponsesAgentOutputItem
                outputs.append(ResponsesAgentOutputItem(
                    type="text",
                    content=content,
                    id=str(uuid4())
                ))
    
    # Return with context for client to persist
    return ResponsesAgentResponse(
        output=outputs,
        custom_outputs={"context": final_context}
    )


@stream()
async def streaming(
    request: ResponsesAgentRequest,
) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
    """
    Streaming endpoint with context persistence.
    
    CRITICAL: Emits __ORBIT_CONTEXT_UPDATE__ at the end for client to capture.
    """
    agent = await init_agent()
    cc_msgs = to_chat_completions_input([i.model_dump() for i in request.input])
    
    # Extract initial context
    initial_context = request.custom_inputs.get("context", {}) if request.custom_inputs else {}
    
    # Track latest context
    latest_context = initial_context
    seen_ids: set[str] = set()
    first_message = True
    
    def get_msg_id(msg):
        if hasattr(msg, 'id'):
            return msg.id
        elif isinstance(msg, dict):
            return msg.get('id', id(msg))
        else:
            return id(msg)
    
    # Stream with both updates and values to capture context
    async for chunk_type, events in agent.astream(
        {"messages": cc_msgs, "context": initial_context},
        stream_mode=["updates", "values"]
    ):
        # Capture context updates
        if chunk_type == "__values__":
            if isinstance(events, dict) and "context" in events:
                latest_context = events["context"]
            continue
        
        # Process message updates
        new_msgs = []
        for node_name, node_data in events.items():
            # Handle node_data which might be dict or other type
            if isinstance(node_data, dict):
                msgs = node_data.get("messages", [])
                # Ensure msgs is a list
                if not isinstance(msgs, list):
                    msgs = [msgs]
                for msg in msgs:
                    if get_msg_id(msg) not in seen_ids:
                        new_msgs.append(msg)
        
        if first_message:
            seen_ids.update(get_msg_id(msg) for msg in new_msgs[:len(cc_msgs)])
            new_msgs = new_msgs[len(cc_msgs):]
            first_message = False
        else:
            seen_ids.update(get_msg_id(msg) for msg in new_msgs)
            node_name = tuple(events.keys())[0] if events else ""
            if node_name in [REASONING, SUPERVISOR]:
                yield ResponsesAgentStreamEvent(
                    type="response.output_item.done",
                    item=ResponsesAgentOutputItem(
                        type="text",
                        content=f"<n>{node_name}</n>",
                        id=str(uuid4())
                    ),
                )
        
        if len(new_msgs) > 0:
            async for event in output_to_responses_items_stream(new_msgs):
                yield event
    
    # CRITICAL: Emit context so client can persist it
    context_json = json.dumps(latest_context)
    yield ResponsesAgentStreamEvent(
        type="response.output_item.done",
        item=ResponsesAgentOutputItem(
            type="text",
            content=f"__ORBIT_CONTEXT_UPDATE__{context_json}",
            id=str(uuid4())
        ),
    )