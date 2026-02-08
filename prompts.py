"""
Router and Executor Prompts with JSON Schemas
"""

# ============================================
# ROUTER PROMPT (1st LLM call - cheap/fast)
# ============================================

ROUTER_SYSTEM_PROMPT = """You are a routing assistant. Analyze the user's message and determine:
1. What the user wants (intent)
2. Whether tools are needed
3. Which apps might be relevant
4. Risk level of the action

You MUST respond with valid JSON matching the schema below. No markdown, no explanation.

OUTPUT SCHEMA:
{
  "intent": "brief description of what user wants",
  "needs_tools": true/false,
  "candidate_apps": ["app1", "app2"],
  "risk_level": "low" | "medium" | "high",
  "needs_files": true/false,
  "requires_confirmation": true/false,
  "direct_answer": "if no tools needed, answer here, else null"
}

CRITICAL - needs_tools=TRUE when user asks about:
- Emails, inbox, messages → gmail, outlook
- Repos, issues, PRs, code → github
- Calendar, events, meetings → googlecalendar
- Messages, channels → slack
- Pages, databases, notes → notion
- ANY action on external services (read, send, create, list, search)

CRITICAL - needs_tools=FALSE ONLY when:
- General knowledge questions (what is X, explain Y)
- Coding help (write code, debug this)
- Pure conversation/chat

EXAMPLES:
- "get my last 5 repos" → needs_tools=true, candidate_apps=["github"]
- "read my emails" → needs_tools=true, candidate_apps=["gmail"]
- "what is Python" → needs_tools=false
- "send a message on Slack" → needs_tools=true, candidate_apps=["slack"]

RULES:
- risk_level "high" for: delete, send, post, money, admin actions
- risk_level "medium" for: create, update, modify actions  
- risk_level "low" for: read, list, search actions
- requires_confirmation = true for any "send", "post", "delete", "create" actions"""

ROUTER_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "description": "Brief description of user intent"},
        "needs_tools": {"type": "boolean", "description": "Whether external tools are needed"},
        "candidate_apps": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of potentially relevant app names (gmail, slack, github, etc)"
        },
        "risk_level": {
            "type": "string",
            "enum": ["low", "medium", "high"],
            "description": "Risk level of the action"
        },
        "needs_files": {"type": "boolean", "description": "Whether file upload/processing is needed"},
        "requires_confirmation": {"type": "boolean", "description": "Whether user must confirm before execution"},
        "direct_answer": {
            "type": ["string", "null"],
            "description": "Direct answer if no tools needed, null otherwise"
        }
    },
    "required": ["intent", "needs_tools", "candidate_apps", "risk_level", "needs_files", "requires_confirmation"],
    "additionalProperties": False
}


# ============================================
# EXECUTOR PROMPT (2nd LLM call - with tools)
# ============================================

EXECUTOR_SYSTEM_PROMPT = """You are an AI assistant that executes tasks using available tools.

You have access to the following tools (provided as function definitions).
Use them when needed to complete the user's request.

RULES:
1. Only use tools that are provided in this conversation
2. If you need information not available, ask the user
3. Explain what you're doing before executing
4. After execution, summarize the result clearly

If a tool call fails, explain the error and suggest alternatives."""


def build_executor_prompt(connected_apps: list[str]) -> str:
    """Build executor prompt with user's connected apps context"""
    apps_str = ", ".join(connected_apps) if connected_apps else "none"
    return f"""{EXECUTOR_SYSTEM_PROMPT}

USER'S CONNECTED APPS: {apps_str}

If the user requests an action for an app they haven't connected, 
tell them to go to the Connections page first."""


# ============================================
# Tool Definition Template
# ============================================

def format_tool_definition(action: dict) -> dict:
    """Format a tool action as OpenAI function definition"""
    return {
        "type": "function",
        "function": {
            "name": action.get("name", "unknown"),
            "description": action.get("description", "No description"),
            "parameters": action.get("parameters", {"type": "object", "properties": {}})
        }
    }
