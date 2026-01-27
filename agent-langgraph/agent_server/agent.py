"""
Orbit Multi-Agent System - Databricks App Deployment
Production-ready pharmaceutical intelligence agent with multi-turn conversation support.
"""

from typing import Annotated, Generator, Literal, Optional, TypedDict
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
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
    output_to_responses_items_stream,
    to_chat_completions_input,
)
from pydantic import BaseModel

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

GENIE = "genie"
SUPERVISOR = "supervisor"
REASONING = "reasoning_agent"
CLARIFICATION = "clarification_agent"

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


# ==================== CONFIGURATION MODELS ====================

class ServedSubAgent(BaseModel):
    """Configuration for a served sub-agent endpoint."""
    endpoint_name: str
    name: str
    task: Literal["agent/v1/responses", "agent/v1/chat", "agent/v2/chat"]
    description: str


class Genie(BaseModel):
    """Configuration for a Genie space agent."""
    space_id: str
    name: str
    task: str = GENIE
    description: str


# Genie agent configurations
GENIE_AGENTS = [
    Genie(
        space_id=GENIE_SPACES["sales"],
        name="sales_agent",
        description="Answers questions about sales performance, revenue trends, primary/secondary sales, and target achievement.",
    ),
    Genie(
        space_id=GENIE_SPACES["stock"],
        name="stock_agent",
        description="Answers questions about inventory levels, stockout risks, backorders, and revenue opportunities.",
    ),
    Genie(
        space_id=GENIE_SPACES["reps"],
        name="reps_agent",
        description="Answers questions about sales rep activity, team performance, call metrics, and coverage.",
    ),
    Genie(
        space_id=GENIE_SPACES["fallback"],
        name="fallback_agent",
        description="Answers complex queries that span multiple domains (sales + stock + reps).",
    ),
]

SERVED_AGENTS: list[ServedSubAgent] = [
    ServedSubAgent(
        endpoint_name=LLM_ENDPOINT_NAME,
        name="reasoning_agent",
        task="agent/v1/chat",
        description="INTERNAL ONLY. Use for synthesizing insights and combining findings.",
    ),
    ServedSubAgent(
        endpoint_name=LLM_ENDPOINT_NAME,
        name="clarification_agent",
        task="agent/v1/chat",
        description="INTERNAL ONLY. Use for asking clarifying questions when queries are ambiguous.",
    ),
]


# ==================== CONVERSATION CONTEXT STATE ====================

class ConversationContext(TypedDict, total=False):
    """Extended state for multi-turn conversation tracking."""
    # Conversation metadata
    conversation_id: str
    turn_count: int
    last_query_time: str

    # Query context tracking
    last_intent: Optional[str]  # "sales_query", "stock_query", "rep_query", "ambiguous"
    last_agent_used: Optional[str]  # Which Genie agent was last used
    waiting_for_clarification: bool  # Is the system waiting for user to clarify?
    clarification_context: Optional[dict]  # What was unclear in the last query

    # Entity tracking (for context carryover)
    current_product: Optional[str]
    current_customer: Optional[str]
    current_region: Optional[str]
    current_rep: Optional[str]
    current_time_period: Optional[str]  # "last month", "Q1 2025", etc.

    # Query refinement history
    query_history: list  # Last 5 queries for context
    refinement_suggestions: list  # Suggestions offered to user


class OrbitState(MessagesState):
    """Extended state with conversation context."""
    context: ConversationContext


# ==================== HELPER FUNCTIONS ====================

def get_msg_attr(msg, attr, default=None):
    """Get message attribute, handling both dict and object types."""
    if hasattr(msg, attr):
        return getattr(msg, attr)
    elif isinstance(msg, dict):
        return msg.get(attr, default)
    return default


def extract_entities(query: str, context: ConversationContext, llm) -> dict:
    """
    Extract pharmaceutical entities from query using LLM.
    Carries over context from previous turns if not explicitly overridden.
    """

    prompt = f"""Extract pharmaceutical business entities from this query.

Query: "{query}"

Previous Context:
- Product: {context.get('current_product', 'None')}
- Customer: {context.get('current_customer', 'None')}
- Region: {context.get('current_region', 'None')}
- Rep: {context.get('current_rep', 'None')}
- Time Period: {context.get('current_time_period', 'None')}

Extract and respond ONLY with a JSON object:
{{
    "product": "product name or code or null",
    "customer": "customer name or null",
    "region": "region name or null",
    "rep": "rep name or code or null",
    "time_period": "time period mentioned or null",
    "carry_over": {{
        "product": true/false,
        "customer": true/false,
        "region": true/false,
        "rep": true/false,
        "time_period": true/false
    }}
}}

Rules:
1. If entity mentioned explicitly → extract it
2. If entity not mentioned but was in previous context → set carry_over=true
3. If entity not mentioned and not in context → set to null
4. Time period examples: "last month", "Q1 2025", "yesterday", "this week"

JSON:"""

    try:
        response = llm.invoke([{"role": "user", "content": prompt}])
        entities = json.loads(response.content.strip())

        # Apply carryover logic
        result = {}
        for key in ["product", "customer", "region", "rep", "time_period"]:
            if entities.get(key):
                result[key] = entities[key]
            elif entities.get("carry_over", {}).get(key):
                result[key] = context.get(f"current_{key}")
            else:
                result[key] = None

        return result
    except Exception as e:
        logger.warning(f"Entity extraction failed: {e}")
        # Fallback to context preservation
        return {
            "product": context.get("current_product"),
            "customer": context.get("current_customer"),
            "region": context.get("current_region"),
            "rep": context.get("current_rep"),
            "time_period": context.get("current_time_period"),
        }


def is_ambiguous_query(query: str, context: ConversationContext, llm) -> tuple[bool, Optional[dict]]:
    """
    Determine if query is too ambiguous and needs clarification.
    Returns: (is_ambiguous, clarification_details)
    """

    prompt = f"""Analyze if this pharmaceutical query needs clarification.

Query: "{query}"

Previous Context:
- Last Intent: {context.get('last_intent', 'None')}
- Product: {context.get('current_product', 'None')}
- Customer: {context.get('current_customer', 'None')}
- Region: {context.get('current_region', 'None')}
- Time Period: {context.get('current_time_period', 'None')}

Determine if the query is too vague to answer effectively.

Examples of AMBIGUOUS queries:
- "Show me sales" (which products? which period?)
- "How are we doing?" (which metric? which entity?)
- "What about stock?" (which products? which customers?)
- "Tell me more" (about what specifically?)

Examples of CLEAR queries:
- "Show me sales for Pantoprazole last month"
- "What's the DSOH for all products in Western Cape?"
- "How are we doing?" (if previous context has product/region)

Respond ONLY with JSON:
{{
    "is_ambiguous": true/false,
    "missing_info": ["product", "time_period", "metric", etc.],
    "can_infer_from_context": true/false,
    "suggested_questions": ["Question 1?", "Question 2?"]
}}

JSON:"""

    try:
        response = llm.invoke([{"role": "user", "content": prompt}])
        result = json.loads(response.content.strip())

        # If can infer from context, not ambiguous
        if result.get("can_infer_from_context"):
            return False, None

        is_ambiguous = result.get("is_ambiguous", False)
        if is_ambiguous:
            return True, {
                "missing_info": result.get("missing_info", []),
                "suggested_questions": result.get("suggested_questions", [])
            }

        return False, None
    except Exception as e:
        logger.warning(f"Ambiguity detection failed: {e}")
        # Default: if very short query with no context, consider ambiguous
        if len(query.split()) < 3 and not context.get("current_product"):
            return True, {"missing_info": ["specific details"], "suggested_questions": []}
        return False, None


# ==================== SUPERVISOR CREATION ====================

def create_langgraph_supervisor(llm, genie_agents, served_agents):
    """
    Create the LangGraph supervisor with multi-turn refinement.

    Features:
    - Conversation context tracking
    - Entity carryover between turns
    - Ambiguity detection and clarification
    - Query refinement suggestions
    - Smart routing based on conversational state
    """

    # Initialize Genie agents
    genie_agent_map = {}
    for agent_config in genie_agents:
        genie_agent = GenieAgent(
            genie_space_id=agent_config.space_id,
            genie_agent_name=agent_config.name,
            description=agent_config.description,
        )
        genie_agent_map[agent_config.name] = genie_agent

    # Initialize reasoning agent
    reasoning_model = ChatDatabricks(endpoint=served_agents[0].endpoint_name)
    reasoning_agent = create_react_agent(
        model=reasoning_model,
        tools=[],
        name=REASONING,
    )

    # Initialize clarification agent
    clarification_model = ChatDatabricks(endpoint=served_agents[1].endpoint_name)
    clarification_agent = create_react_agent(
        model=clarification_model,
        tools=[],
        name=CLARIFICATION,
    )

    # Create agent descriptions for supervisor prompt
    agent_descriptions = "\n".join([
        f"- **{config.name}**: {config.description}"
        for config in genie_agents
    ])

    # Define supervisor routing logic
    def supervisor_node(state: OrbitState) -> Command[Literal[
        "sales_agent", "stock_agent", "reps_agent", "fallback_agent",
        "reasoning_agent", "clarification_agent", "__end__"
    ]]:
        """
        Supervisor with multi-turn conversation awareness.
        """

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
                "clarification_context": None,
            }

        # Check if last message came from a Genie agent
        genie_agent_names = ["sales_agent", "stock_agent", "reps_agent", "fallback_agent"]
        msg_name = get_msg_attr(last_message, 'name')

        if msg_name in genie_agent_names:
            # Genie agent just responded - route to reasoning for synthesis
            context["last_agent_used"] = msg_name
            context["waiting_for_clarification"] = False

            return Command(
                goto=REASONING,
                update={
                    "context": context,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are the reasoning and synthesis agent for Orbit. "
                                "A Genie agent has just provided raw data. Your job is to:\n"
                                "1. Analyze the data provided\n"
                                "2. Extract key insights and patterns\n"
                                "3. Provide a clear, concise business answer\n"
                                "4. Include specific numbers when available\n"
                                "5. Suggest relevant follow-up questions the user might want to explore\n\n"
                                f"Conversation Context:\n"
                                f"- Turn: {context['turn_count']}\n"
                                f"- Product Focus: {context.get('current_product', 'None')}\n"
                                f"- Region Focus: {context.get('current_region', 'None')}\n\n"
                                "IMPORTANT: Never mention internal agents. Present as Orbit."
                            )
                        }
                    ]
                }
            )

        # Check if reasoning agent just responded
        if msg_name == REASONING:
            # Synthesis complete - end conversation (wait for next user input)
            return Command(
                goto=END,
                update={"context": context}
            )

        # Check if clarification agent just responded
        if msg_name == CLARIFICATION:
            # Clarification question sent - end turn and wait for user response
            context["waiting_for_clarification"] = True
            return Command(
                goto=END,
                update={"context": context}
            )

        # Otherwise, supervisor needs to route the user query
        context["turn_count"] = context.get("turn_count", 0) + 1
        context["last_query_time"] = datetime.now().isoformat()

        user_query = get_msg_attr(last_message, 'content', str(last_message))

        # Extract entities from query (with context carryover)
        entities = extract_entities(user_query, context, llm)

        # Update context with extracted entities
        context.update({
            "current_product": entities.get("product") or context.get("current_product"),
            "current_customer": entities.get("customer") or context.get("current_customer"),
            "current_region": entities.get("region") or context.get("current_region"),
            "current_rep": entities.get("rep") or context.get("current_rep"),
            "current_time_period": entities.get("time_period") or context.get("current_time_period"),
        })

        # Add to query history
        query_history = context.get("query_history", [])
        query_history.append({
            "turn": context["turn_count"],
            "query": user_query,
            "entities": entities,
            "timestamp": context["last_query_time"]
        })
        # Keep only last 5 queries
        context["query_history"] = query_history[-5:]

        # Check if query is ambiguous
        is_ambiguous, clarification_details = is_ambiguous_query(user_query, context, llm)

        if is_ambiguous and not context.get("waiting_for_clarification"):
            # Query is too vague - ask for clarification
            context["clarification_context"] = clarification_details

            return Command(
                goto=CLARIFICATION,
                update={
                    "context": context,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are asking for clarification on an ambiguous query. "
                                "The user asked a vague question. Help them refine it.\n\n"
                                f"User Query: {user_query}\n"
                                f"Missing Info: {clarification_details.get('missing_info', [])}\n"
                                f"Suggested Questions: {clarification_details.get('suggested_questions', [])}\n\n"
                                "Your response should:\n"
                                "1. Acknowledge their question\n"
                                "2. Politely explain what additional info would help\n"
                                "3. Provide 2-3 specific example questions they could ask\n"
                                "4. Be friendly and helpful, not robotic\n\n"
                                "Example:\n"
                                "\"I'd be happy to help! To give you the most relevant information, "
                                "could you let me know which products or time period you're interested in? "
                                "For example:\n"
                                "- Sales for Pantoprazole in December 2025\n"
                                "- Stock levels for Pain category in Western Cape\n"
                                "- Rep performance for the Cape Town team last week\""
                            )
                        }
                    ]
                }
            )

        # Query is clear - route to appropriate agent
        system_prompt = f"""You are the Orbit Supervisor routing pharmaceutical intelligence queries.

## CONVERSATION CONTEXT
- Turn: {context['turn_count']}
- Product: {context.get('current_product', 'Not specified')}
- Customer: {context.get('current_customer', 'Not specified')}
- Region: {context.get('current_region', 'Not specified')}
- Rep: {context.get('current_rep', 'Not specified')}
- Time Period: {context.get('current_time_period', 'Not specified')}
- Last Agent: {context.get('last_agent_used', 'None')}

## RECENT QUERIES
{chr(10).join([f"Turn {q['turn']}: {q['query']}" for q in context.get('query_history', [])[-3:]])}

## ROUTING RULES
For greetings/thanks: "DIRECT_RESPONSE:[message]"
For off-topic: "DIRECT_RESPONSE:[polite decline]"
For data questions:
- Sales/revenue/targets → "ROUTE:sales_agent"
- Stock/inventory/stockout → "ROUTE:stock_agent"
- Rep/team/calls → "ROUTE:reps_agent"
- Complex/multi-domain → "ROUTE:fallback_agent"

## CURRENT QUERY
"{user_query}"

DECISION (one line only):"""

        # Get supervisor decision
        try:
            response = llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ])

            decision = response.content.strip()

            logger.info(f"Turn {context['turn_count']} | Decision: {decision}")

            # Parse decision
            if decision.startswith("ROUTE:"):
                target_agent = decision.replace("ROUTE:", "").strip()
                intent = target_agent.replace("_agent", "_query")
                context["last_intent"] = intent

                if target_agent in genie_agent_names:
                    return Command(
                        goto=target_agent,
                        update={"context": context}
                    )
                else:
                    # Invalid route, default to fallback
                    return Command(
                        goto="fallback_agent",
                        update={"context": context}
                    )

            elif decision.startswith("DIRECT_RESPONSE:"):
                response_text = decision.replace("DIRECT_RESPONSE:", "").strip()
                return Command(
                    goto=END,
                    update={
                        "context": context,
                        "messages": [{"role": "assistant", "content": response_text}]
                    }
                )

            else:
                # Couldn't parse decision, default to fallback agent
                logger.warning(f"Couldn't parse decision: {decision}")
                context["last_intent"] = "fallback_query"
                return Command(
                    goto="fallback_agent",
                    update={"context": context}
                )
        except Exception as e:
            logger.error(f"Routing error: {e}", exc_info=True)
            # Error fallback
            return Command(
                goto="fallback_agent",
                update={"context": context}
            )

    # Build the graph
    workflow = StateGraph(OrbitState)

    # Add supervisor node
    workflow.add_node(SUPERVISOR, supervisor_node)

    # Add Genie agent nodes
    for name, agent in genie_agent_map.items():
        workflow.add_node(name, agent)

    # Add reasoning agent node
    workflow.add_node(REASONING, reasoning_agent)

    # Add clarification agent node
    workflow.add_node(CLARIFICATION, clarification_agent)

    # Define edges
    workflow.add_edge(START, SUPERVISOR)

    # All agents flow back to supervisor
    for name in genie_agent_map.keys():
        workflow.add_edge(name, SUPERVISOR)

    workflow.add_edge(REASONING, SUPERVISOR)
    workflow.add_edge(CLARIFICATION, SUPERVISOR)

    return workflow.compile()


# ==================== GLOBAL AGENT INSTANCE ====================

_supervisor_agent = None


def get_supervisor():
    """Get or create the supervisor agent singleton."""
    global _supervisor_agent
    if _supervisor_agent is None:
        llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)
        _supervisor_agent = create_langgraph_supervisor(llm, GENIE_AGENTS, SERVED_AGENTS)
    return _supervisor_agent


# ==================== MLFLOW RESPONSES AGENT ====================

class OrbitResponsesAgent(ResponsesAgent):
    """
    Wraps the LangGraph supervisor as a ResponsesAgent for MLflow serving.
    Handles multi-turn conversations with context persistence.
    """

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        """Synchronous prediction that collects all streamed outputs."""
        outputs = []
        for event in self.predict_stream(request):
            if event.type == "response.output_item.done":
                outputs.append(event.item)
        return ResponsesAgentResponse(output=outputs, custom_outputs=request.custom_inputs)

    def predict_stream(
        self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        """Streaming prediction that yields events as they occur."""
        agent = get_supervisor()
        cc_msgs = to_chat_completions_input([i.model_dump() for i in request.input])

        # Extract conversation context from custom_inputs if available
        context = request.custom_inputs.get("context", {}) if request.custom_inputs else {}

        first_message = True
        seen_ids: set[str] = set()

        for _, events in agent.stream(
            {"messages": cc_msgs, "context": context},
            stream_mode=["updates"]
        ):
            # Helper to get message ID (handles both dict and object types)
            def get_msg_id(msg):
                if hasattr(msg, 'id'):
                    return msg.id
                elif isinstance(msg, dict):
                    return msg.get('id', id(msg))  # Use Python id() as fallback
                else:
                    return id(msg)

            new_msgs = [
                msg
                for v in events.values()
                for msg in v.get("messages", [])
                if get_msg_id(msg) not in seen_ids
            ]
            if first_message:
                # Skip the initial input messages on first iteration
                seen_ids.update(get_msg_id(msg) for msg in new_msgs[:len(cc_msgs)])
                new_msgs = new_msgs[len(cc_msgs):]
                first_message = False
            else:
                seen_ids.update(get_msg_id(msg) for msg in new_msgs)
                # Emit node name as a marker (useful for debugging)
                node_name = tuple(events.keys())[0] if events else ""
                # Only show reasoning/clarification agent output to user
                if node_name in [REASONING, CLARIFICATION, SUPERVISOR]:
                    yield ResponsesAgentStreamEvent(
                        type="response.output_item.done",
                        item=self.create_text_output_item(text=f"<n>{node_name}</n>", id=str(uuid4())),
                    )
            if len(new_msgs) > 0:
                yield from output_to_responses_items_stream(new_msgs)


# Register with MLflow - instantiate the agent
mlflow.models.set_model(OrbitResponsesAgent())
