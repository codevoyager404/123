"""
Worker Jobs for Background Processing
Uses RQ (Redis Queue) for async task processing
"""
import json
import hashlib
import time
from datetime import datetime
from typing import Dict, Any, Optional
from redis import Redis
from rq import Queue
from openai import OpenAI

from config import (
    OPENAI_API_KEY, ROUTER_MODEL, EXECUTOR_MODEL,
    REDIS_URL, MAX_TOOL_CALLS_PER_TASK, MAX_RETRIES,
    ROUTER_PROMPT_VERSION, EXECUTOR_PROMPT_VERSION,
    CONFIRM_TTL_MINUTES, CONFIRM_TTL_SEND_MINUTES
)
from prompts import ROUTER_SYSTEM_PROMPT, ROUTER_JSON_SCHEMA, build_executor_prompt, format_tool_definition
import database as db


# Redis connection
redis_conn = Redis.from_url(REDIS_URL)
task_queue = Queue("tasks", connection=redis_conn)


# OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)


# ============================================
# Helper Functions
# ============================================

def generate_idempotency_key(task_id: str, step_id: str, tool_name: str, params: Dict) -> str:
    """Generate unique idempotency key for tool execution"""
    normalized = json.dumps(params, sort_keys=True)
    content = f"{task_id}:{step_id}:{tool_name}:{normalized}"
    return hashlib.sha256(content.encode()).hexdigest()[:32]


def is_transient_error(error_code: str) -> bool:
    """Check if error is transient (retryable)"""
    transient_codes = ["timeout", "rate_limit", "5xx", "connection_error", "503", "502", "504"]
    return any(code in error_code.lower() for code in transient_codes)


def publish_event(task_id: str, event_type: str, data: Dict):
    """Publish SSE event to Redis pubsub"""
    event = {
        "type": event_type,
        "task_id": task_id,
        "data": data,
        "timestamp": datetime.utcnow().isoformat()
    }
    redis_conn.publish(f"task:{task_id}", json.dumps(event))


# ============================================
# Main Task Processor
# ============================================

def process_task(task_id: str, user_id: str, message: str, conversation_history: list):
    """
    Main worker job - processes a chat message through the full pipeline:
    Router → Tool Search → Executor → Response
    """
    import asyncio
    
    async def _process():
        # Update task to running
        await db.update_task(task_id, status=db.TaskStatus.RUNNING)
        publish_event(task_id, "status", {"status": "running"})
        
        try:
            # ============================================
            # Step 1: ROUTER (cheap LLM call)
            # ============================================
            publish_event(task_id, "progress", {"step": "router", "message": "Analyzing request..."})
            
            router_messages = [
                {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": message}
            ]
            
            router_response = client.chat.completions.create(
                model=ROUTER_MODEL,
                messages=router_messages,
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=500
            )
            
            router_output = json.loads(router_response.choices[0].message.content)
            
            # Save router output
            await db.update_task(
                task_id,
                router_output=router_output,
                router_prompt_version=ROUTER_PROMPT_VERSION
            )
            
            publish_event(task_id, "router", {"output": router_output})
            
            # ============================================
            # Step 2: Check if tools needed
            # ============================================
            if not router_output.get("needs_tools", False):
                # No tools needed - return direct answer
                direct_answer = router_output.get("direct_answer") or "I can help with that! However, I wasn't able to generate a specific response."
                
                await db.update_task(
                    task_id,
                    status=db.TaskStatus.SUCCEEDED,
                    result={"response": direct_answer}
                )
                
                publish_event(task_id, "final", {"response": direct_answer})
                return {"success": True, "response": direct_answer}
            
            # ============================================
            # Step 3: Policy Gate + Tool Search
            # ============================================
            publish_event(task_id, "progress", {"step": "tool_search", "message": "Finding relevant tools..."})
            
            # Get user's connected apps
            connected_apps = await db.get_user_connections(user_id)
            
            candidate_apps = router_output.get("candidate_apps", [])
            risk_level = router_output.get("risk_level", "low")
            requires_confirmation = router_output.get("requires_confirmation", False)
            
            # Policy gate: filter to only connected apps
            allowed_apps = [app for app in candidate_apps if app.lower() in [c.lower() for c in connected_apps]]
            
            if not allowed_apps and candidate_apps:
                # User hasn't connected required apps
                missing = [app for app in candidate_apps if app.lower() not in [c.lower() for c in connected_apps]]
                error_msg = f"To complete this action, please connect the following apps first: {', '.join(missing)}. Go to the Connections page to set this up."
                
                await db.update_task(
                    task_id,
                    status=db.TaskStatus.FAILED,
                    error=error_msg
                )
                
                publish_event(task_id, "final", {"response": error_msg, "error": True})
                return {"success": False, "error": error_msg}
            
            # Get Top-N tools from catalog (simplified - would query tool_actions table)
            # For MVP, we'll use placeholder tools based on app names
            selected_tools = get_tools_for_apps(allowed_apps, limit=8)
            
            publish_event(task_id, "tools_selected", {"tools": [t["function"]["name"] for t in selected_tools]})
            
            # ============================================
            # Step 4: Check if confirmation needed
            # ============================================
            if requires_confirmation:
                ttl = CONFIRM_TTL_SEND_MINUTES if "send" in router_output.get("intent", "").lower() else CONFIRM_TTL_MINUTES
                confirm_token = await db.set_task_awaiting_confirm(task_id, ttl_minutes=ttl)
                
                preview = {
                    "intent": router_output.get("intent"),
                    "apps_involved": allowed_apps,
                    "risk_level": risk_level,
                    "confirm_token": confirm_token
                }
                
                publish_event(task_id, "awaiting_confirm", preview)
                return {"success": True, "awaiting_confirm": True, "preview": preview}
            
            # ============================================
            # Step 5: EXECUTOR (LLM with tools)
            # ============================================
            return await execute_with_tools(
                task_id, user_id, message, conversation_history,
                selected_tools, connected_apps
            )
            
        except Exception as e:
            error_msg = str(e)
            await db.update_task(
                task_id,
                status=db.TaskStatus.FAILED,
                error=error_msg
            )
            publish_event(task_id, "error", {"message": error_msg})
            return {"success": False, "error": error_msg}
    
    # Run async code
    return asyncio.run(_process())


async def execute_with_tools(
    task_id: str,
    user_id: str,
    message: str,
    conversation_history: list,
    tools: list,
    connected_apps: list
) -> Dict[str, Any]:
    """Execute the task with tools"""
    from composio_client import execute_composio_action
    
    publish_event(task_id, "progress", {"step": "executor", "message": "Executing with tools..."})
    
    # Build messages
    messages = [
        {"role": "system", "content": build_executor_prompt(connected_apps)}
    ]
    
    # Add conversation history (last N messages)
    for msg in conversation_history[-10:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    
    messages.append({"role": "user", "content": message})
    
    # Call LLM with tools
    response = client.chat.completions.create(
        model=EXECUTOR_MODEL,
        messages=messages,
        tools=tools if tools else None,
        temperature=0.7,
        max_tokens=2000
    )
    
    assistant_message = response.choices[0].message
    tool_calls_count = 0
    
    # Handle tool calls
    while assistant_message.tool_calls and tool_calls_count < MAX_TOOL_CALLS_PER_TASK:
        tool_results = []
        
        for tool_call in assistant_message.tool_calls:
            tool_calls_count += 1
            tool_name = tool_call.function.name
            tool_params = json.loads(tool_call.function.arguments)
            
            # Create task step
            idempotency_key = generate_idempotency_key(task_id, tool_call.id, tool_name, tool_params)
            step_id = await db.create_task_step(task_id, tool_calls_count, tool_name, tool_params, idempotency_key)
            
            publish_event(task_id, "tool_call", {
                "tool": tool_name,
                "params": tool_params,
                "step": tool_calls_count
            })
            
            # Execute tool with retry logic
            result, error = await execute_with_retry(
                user_id, tool_name, tool_params, step_id
            )
            
            if error:
                await db.update_task_step(step_id, status="failed", last_error_code=error)
                tool_results.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "content": json.dumps({"error": error})
                })
            else:
                await db.update_task_step(step_id, status="succeeded", tool_result=result)
                tool_results.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "content": json.dumps(result)
                })
            
            publish_event(task_id, "tool_result", {
                "tool": tool_name,
                "success": error is None
            })
        
        # Continue conversation with tool results
        messages.append(assistant_message)
        messages.extend(tool_results)
        
        response = client.chat.completions.create(
            model=EXECUTOR_MODEL,
            messages=messages,
            tools=tools,
            temperature=0.7,
            max_tokens=2000
        )
        
        assistant_message = response.choices[0].message
    
    # Final response
    final_response = assistant_message.content or "Task completed."
    
    await db.update_task(
        task_id,
        status=db.TaskStatus.SUCCEEDED,
        executor_prompt_version=EXECUTOR_PROMPT_VERSION,
        result={"response": final_response}
    )
    
    publish_event(task_id, "final", {"response": final_response})
    
    return {"success": True, "response": final_response}


async def execute_with_retry(
    user_id: str,
    tool_name: str,
    tool_params: Dict,
    step_id: str,
    max_retries: int = MAX_RETRIES
) -> tuple[Optional[Dict], Optional[str]]:
    """Execute a tool call with retry logic"""
    from composio_client import execute_composio_action
    
    entity_id = f"user_{user_id}"
    attempt = 0
    last_error = None
    
    while attempt <= max_retries:
        try:
            await db.update_task_step(step_id, attempt=attempt)
            
            result = await execute_composio_action(tool_name, entity_id, tool_params)
            
            if "error" in result:
                error_code = result.get("error", "unknown")
                
                if is_transient_error(error_code) and attempt < max_retries:
                    # Exponential backoff with jitter
                    delay = (2 ** attempt) + (time.time() % 1)
                    time.sleep(delay)
                    attempt += 1
                    last_error = error_code
                    continue
                else:
                    return None, error_code
            
            return result, None
            
        except Exception as e:
            error_code = str(e)
            
            if is_transient_error(error_code) and attempt < max_retries:
                delay = (2 ** attempt) + (time.time() % 1)
                time.sleep(delay)
                attempt += 1
                last_error = error_code
            else:
                return None, error_code
    
    return None, last_error or "max_retries_exceeded"


def get_tools_for_apps(apps: list, limit: int = 8) -> list:
    """Get tool definitions for given apps (simplified MVP version)"""
    # In production, this would query the tool_actions table
    # For MVP, we return common actions for known apps
    
    tool_definitions = {
        "gmail": [
            {
                "type": "function",
                "function": {
                    "name": "GMAIL_SEND_EMAIL",
                    "description": "Send an email via Gmail",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "to": {"type": "string", "description": "Recipient email"},
                            "subject": {"type": "string", "description": "Email subject"},
                            "body": {"type": "string", "description": "Email body"}
                        },
                        "required": ["to", "subject", "body"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "GMAIL_FETCH_EMAILS",
                    "description": "Fetch recent emails from Gmail",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "max_results": {"type": "integer", "description": "Max emails to fetch", "default": 10}
                        }
                    }
                }
            }
        ],
        "slack": [
            {
                "type": "function",
                "function": {
                    "name": "SLACK_SEND_MESSAGE",
                    "description": "Send a message to a Slack channel",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "channel": {"type": "string", "description": "Channel name"},
                            "text": {"type": "string", "description": "Message text"}
                        },
                        "required": ["channel", "text"]
                    }
                }
            }
        ],
        "github": [
            {
                "type": "function",
                "function": {
                    "name": "GITHUB_CREATE_ISSUE",
                    "description": "Create an issue in a GitHub repository",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "owner": {"type": "string", "description": "Repository owner"},
                            "repo": {"type": "string", "description": "Repository name"},
                            "title": {"type": "string", "description": "Issue title"},
                            "body": {"type": "string", "description": "Issue body"}
                        },
                        "required": ["owner", "repo", "title"]
                    }
                }
            }
        ],
        "notion": [
            {
                "type": "function",
                "function": {
                    "name": "NOTION_CREATE_PAGE",
                    "description": "Create a new page in Notion",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string", "description": "Page title"},
                            "content": {"type": "string", "description": "Page content"}
                        },
                        "required": ["title"]
                    }
                }
            }
        ]
    }
    
    result = []
    for app in apps:
        app_lower = app.lower()
        if app_lower in tool_definitions:
            result.extend(tool_definitions[app_lower])
        if len(result) >= limit:
            break
    
    return result[:limit]


# ============================================
# Janitor Job (cleanup expired tasks)
# ============================================

def cleanup_expired_tasks():
    """
    Background job to cleanup expired confirmation tasks.
    Should be run periodically (e.g., every 1-5 minutes)
    """
    import asyncio
    
    async def _cleanup():
        expired_tasks = await db.get_expired_tasks()
        
        for task in expired_tasks:
            task_id = task["id"]
            
            await db.update_task(
                task_id,
                status=db.TaskStatus.EXPIRED,
                error="Confirmation timeout expired"
            )
            
            publish_event(task_id, "expired", {
                "message": "Η επιβεβαίωση έληξε. Ξαναζήτησε αν θέλεις να συνεχίσεις."
            })
        
        return len(expired_tasks)
    
    return asyncio.run(_cleanup())


# ============================================
# Queue Helpers
# ============================================

def enqueue_task(task_id: str, user_id: str, message: str, conversation_history: list):
    """Add task to the processing queue"""
    return task_queue.enqueue(
        process_task,
        task_id,
        user_id,
        message,
        conversation_history,
        job_id=f"task_{task_id}",
        job_timeout="10m"
    )
