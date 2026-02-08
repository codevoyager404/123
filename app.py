"""
AI Agent MVP - FastAPI Application
With Composio SDK integration for real tool execution
"""
import os
import json
import secrets
import hashlib
import asyncio
import re
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr
from sse_starlette.sse import EventSourceResponse
import jwt
from openai import OpenAI

# Composio SDK
from composio import Composio
from composio_openai import OpenAIProvider

from config import JWT_SECRET, APP_URL, OPENAI_API_KEY, COMPOSIO_API_KEY, EXECUTOR_MODEL, ROUTER_PROMPT_VERSION
import database as db
from prompts import ROUTER_SYSTEM_PROMPT

# Static files directory
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
# Keep Composio cache inside workspace so it's writable in local/dev sandboxes.
os.environ.setdefault("COMPOSIO_CACHE_DIR", os.path.join(os.path.dirname(__file__), ".composio_cache"))

# OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Composio client with OpenAI provider
composio = Composio(
    api_key=COMPOSIO_API_KEY,
    provider=OpenAIProvider(),
    dangerously_skip_version_check=True,
)

# In-memory event store for SSE
task_events: Dict[str, List[Dict]] = {}


# ============================================
# Lifespan - DB Init
# ============================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.init_db()
    yield


app = FastAPI(title="AI Agent MVP", version="3.2.0", lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# Models
# ============================================

class UserRegister(BaseModel):
    email: EmailStr
    password: str
    name: str


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatSendRequest(BaseModel):
    message: str
    conversation_history: Optional[List[ChatMessage]] = []


class ChatSendResponse(BaseModel):
    task_id: str
    status: str


class TaskConfirmRequest(BaseModel):
    confirm_token: str
    approved: bool


# ============================================
# Auth Helpers
# ============================================

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def create_token(user_id: str) -> str:
    payload = {
        "user_id": user_id,
        "exp": datetime.utcnow() + timedelta(days=7)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


def verify_token(token: str) -> Optional[str]:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        return payload.get("user_id")
    except:
        return None


async def get_current_user(request: Request) -> Optional[Dict]:
    token = request.cookies.get("auth_token")
    
    if not token:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:]
    
    if not token:
        return None
    
    user_id = verify_token(token)
    if user_id:
        return await db.get_user_by_id(user_id)
    return None


# ============================================
# Event Publishing (in-memory)
# ============================================

def publish_event(task_id: str, event_type: str, data: Dict):
    """Store event for SSE streaming"""
    if task_id not in task_events:
        task_events[task_id] = []
    
    event = {
        "type": event_type,
        "data": data,
        "timestamp": datetime.utcnow().isoformat()
    }
    task_events[task_id].append(event)
    print(f"📡 Event [{event_type}]: {json.dumps(data, ensure_ascii=False)[:100]}")


def _normalize_toolkits(toolkits_resp: Any) -> list[Dict[str, Any]]:
    """
    Normalize `session.toolkits()` response into a list of dicts:
    {slug, name, is_active, connected_account_id}
    """
    items = getattr(toolkits_resp, "items", None)
    if items is None:
        items = toolkits_resp if isinstance(toolkits_resp, list) else []

    normalized: list[Dict[str, Any]] = []
    for t in items or []:
        if isinstance(t, dict):
            slug = (t.get("slug") or t.get("key") or t.get("toolkit") or t.get("name") or "").lower()
            name = t.get("name") or slug
            is_active = bool(t.get("connected") or t.get("is_connected") or t.get("authorized") or t.get("is_active"))
            connected_account_id = (
                t.get("connected_account_id")
                or (t.get("connection") or {}).get("connected_account_id")
                or ((t.get("connection") or {}).get("connected_account") or {}).get("id")
            )
        else:
            slug = str(getattr(t, "slug", None) or getattr(t, "key", None) or getattr(t, "toolkit", None) or getattr(t, "name", None) or "").lower()
            name = getattr(t, "name", None) or slug
            conn = getattr(t, "connection", None)
            is_active = bool(getattr(conn, "is_active", False) or getattr(t, "is_active", False))
            connected_account = getattr(conn, "connected_account", None)
            connected_account_id = getattr(connected_account, "id", None) if connected_account else None

        if not slug:
            continue
        normalized.append(
            {
                "slug": slug,
                "name": name,
                "is_active": bool(is_active),
                "connected_account_id": connected_account_id,
            }
        )
    return normalized


def _normalize_connected_accounts(accounts_resp: Any) -> list[Dict[str, Any]]:
    """
    Normalize `composio.connected_accounts.list(...)` response into a list of dicts:
    {id, status, toolkit_slug}
    """
    items = getattr(accounts_resp, "items", None)
    if items is None:
        items = accounts_resp if isinstance(accounts_resp, list) else []

    normalized: list[Dict[str, Any]] = []
    for a in items or []:
        if isinstance(a, dict):
            account_id = a.get("id")
            status = a.get("status")
            toolkit = a.get("toolkit") or {}
            toolkit_slug = (toolkit.get("slug") or toolkit.get("key") or toolkit.get("name") or "").upper()
        else:
            account_id = getattr(a, "id", None)
            status = getattr(a, "status", None)
            toolkit = getattr(a, "toolkit", None)
            toolkit_slug = str(getattr(toolkit, "slug", None) or getattr(toolkit, "key", None) or getattr(toolkit, "name", None) or "").upper()

        if account_id and toolkit_slug:
            normalized.append({"id": account_id, "status": status, "toolkit_slug": toolkit_slug})
    return normalized


def _select_tools_for_message(tools: list[Dict[str, Any]], message: str, limit: int = 12) -> list[Dict[str, Any]]:
    """
    Heuristic tool selection to avoid overwhelming the model and reduce wrong tool choices.
    Prefer read-only tools unless the user explicitly requests a write action.
    """
    if not tools:
        return []

    msg = (message or "").lower()
    msg_tokens = set(re.findall(r"[a-z0-9_]+", msg))

    repos_query = ("repo" in msg) or ("repos" in msg) or ("repository" in msg) or ("repositories" in msg)
    email_query = ("email" in msg) or ("emails" in msg) or ("mail" in msg) or ("inbox" in msg) or ("gmail" in msg) or ("outlook" in msg)
    last_query = ("last" in msg) or ("latest" in msg) or ("recent" in msg) or ("τελευτ" in msg)

    write_hints = {
        "create",
        "update",
        "delete",
        "remove",
        "add",
        "accept",
        "invite",
        "send",
        "post",
        "merge",
        "close",
        "approve",
        "write",
        "edit",
        "σβησε",
        "διαγραψε",
        "στειλε",
        "δημιουργησε",
        "αποδεξου",
        "κανε",
    }
    allow_writes = any(h in msg for h in write_hints)

    dangerous_markers = {
        "CREATE",
        "UPDATE",
        "DELETE",
        "REMOVE",
        "ADD_",
        "ACCEPT",
        "INVITE",
        "SEND",
        "POST",
        "MERGE",
        "CLOSE",
        "APPROVE",
        "ASSIGN",
        "TRANSFER",
        "SET_",
    }

    read_only_markers = {"LIST", "GET", "SEARCH", "FETCH", "READ"}
    repo_allowed_required = {
        "per_page",
        "page",
        "sort",
        "direction",
        "visibility",
        "affiliation",
        "type",
        "since",
        "username",
        "org",
        "owner",
    }
    email_allowed_required = {
        "max_results",
        "query",
        "q",
        "page_token",
        "label_ids",
        "include_spam_trash",
        "format",
    }

    def required_params(tool: Dict[str, Any]) -> set[str]:
        fn = (tool.get("function") or {})
        params = fn.get("parameters") or {}
        req = params.get("required") or []
        return {str(x) for x in req}

    def score(tool: Dict[str, Any]) -> int:
        fn = (tool.get("function") or {})
        name = str(fn.get("name") or "")
        desc = str(fn.get("description") or "")

        # Penalize write-ish tools unless explicitly requested, but don't hard-drop.
        if not allow_writes and any(m in name for m in dangerous_markers):
            base_penalty = -200
        else:
            base_penalty = 0

        name_tokens = set(re.findall(r"[A-Z0-9]+", name.upper()))
        desc_tokens = set(re.findall(r"[a-z0-9_]+", desc.lower()))

        overlap = len({t.upper() for t in msg_tokens} & name_tokens)
        overlap += len(msg_tokens & desc_tokens)

        s = (overlap * 10) + base_penalty

        # Boost common "list repos" queries.
        if "repo" in msg or "repos" in msg or "repository" in msg or "repositories" in msg:
            if "REPOSITORY" in name.upper() and ("LIST" in name.upper() or "GET" in name.upper()):
                s += 50
            if "INVITATION" in name.upper():
                s -= 50

        if "last" in msg or "latest" in msg or "recent" in msg:
            if "LIST" in name.upper() or "GET" in name.upper():
                s += 10

        return s

    ranked = sorted(tools, key=score, reverse=True)

    # Hard preference for common read tasks when possible (pre-filter by name patterns).
    if not allow_writes and repos_query:
        repo_candidates = []
        for t in ranked:
            name_u = str(((t.get("function") or {}).get("name")) or "").upper()
            req = required_params(t)
            if ("REPO" in name_u or "REPOSITORY" in name_u) and any(m in name_u for m in read_only_markers):
                if "INVITATION" in name_u:
                    continue
                if req and any(r not in repo_allowed_required for r in req):
                    continue
                repo_candidates.append(t)
            if len(repo_candidates) >= limit:
                break
        if repo_candidates:
            return repo_candidates
        # Do not allow unrelated tools for repo queries.
        return []

    if not allow_writes and email_query:
        email_candidates = []
        for t in ranked:
            name_u = str(((t.get("function") or {}).get("name")) or "").upper()
            req = required_params(t)
            if any(m in name_u for m in read_only_markers) and any(k in name_u for k in ["EMAIL", "MESSAGE", "MESSAGES", "THREAD", "INBOX"]):
                if req and any(r not in email_allowed_required for r in req):
                    continue
                email_candidates.append(t)
            if len(email_candidates) >= limit:
                break
        if email_candidates:
            return email_candidates
        # Do not allow unrelated tools for email queries.
        return []

    if not allow_writes:
        ro = []
        for t in ranked:
            name = str(((t.get("function") or {}).get("name")) or "").upper()
            if any(m in name for m in read_only_markers):
                ro.append(t)
            if len(ro) >= limit:
                break
        if ro:
            return ro

    return ranked[:limit]


# ============================================
# Task Processing with Composio
# ============================================

async def process_task_with_composio(task_id: str, user_id: str, message: str, conversation_history: list):
    """Process task using Composio SDK for tool execution"""
    
    await db.update_task(task_id, status=db.TaskStatus.RUNNING)
    publish_event(task_id, "status", {"status": "running"})
    
    try:
        # ============================================
        # Step 1: ROUTER (intent detection)
        # ============================================
        publish_event(task_id, "progress", {"step": "router", "message": "Αναλύω το αίτημα..."})
        
        router_messages = [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": message}
        ]
        
        router_response = client.chat.completions.create(
            model="gpt-4o-mini",  # Cheap for routing
            messages=router_messages,
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=500
        )
        
        router_output = json.loads(router_response.choices[0].message.content)
        
        await db.update_task(
            task_id,
            router_output=router_output,
            router_prompt_version=ROUTER_PROMPT_VERSION
        )
        
        publish_event(task_id, "router", {"output": router_output})
        
        # ============================================
        # Step 2: Check if tools needed
        # ============================================
        needs_tools = router_output.get("needs_tools", False)
        candidate_apps = router_output.get("candidate_apps", [])

        # Deterministic hints to reduce router mistakes (e.g. forcing Outlook for generic "emails").
        msg_l = (message or "").lower()
        hinted: list[str] = []
        for k in ["github", "gmail", "outlook", "slack", "notion", "googlecalendar"]:
            if k in msg_l:
                hinted.append(k)

        if hinted:
            seen = set()
            merged: list[str] = []
            for a in (candidate_apps or []) + hinted:
                a_l = str(a).lower()
                if a_l and a_l not in seen:
                    seen.add(a_l)
                    merged.append(a_l)
            candidate_apps = merged
        else:
            # Generic email request: offer both providers if not explicitly specified.
            is_email_request = ("email" in msg_l) or ("emails" in msg_l) or ("mail" in msg_l) or ("e-mail" in msg_l)
            if is_email_request and (not candidate_apps or candidate_apps == ["outlook"] or candidate_apps == ["gmail"]):
                candidate_apps = ["gmail", "outlook"]

        if not needs_tools:
            # Pure chat - no tools needed
            publish_event(task_id, "progress", {"step": "generating", "message": "Γράφω απάντηση..."})
            
            chat_response = client.chat.completions.create(
                model=EXECUTOR_MODEL,
                messages=[
                    {"role": "system", "content": "Είσαι ένας χρήσιμος AI assistant. Απάντησε στα ελληνικά αν ο χρήστης γράφει ελληνικά."},
                    *[{"role": m["role"], "content": m["content"]} for m in conversation_history[-5:]],
                    {"role": "user", "content": message}
                ],
                temperature=0.7,
                max_tokens=1500
            )
            
            response_text = chat_response.choices[0].message.content
            
            await db.update_task(
                task_id,
                status=db.TaskStatus.SUCCEEDED,
                result={"response": response_text}
            )
            
            publish_event(task_id, "final", {"response": response_text})
            return
        
        # ============================================
        # Step 3: Tools needed - Use Composio
        # ============================================
        publish_event(
            task_id,
            "progress",
            {"step": "tools", "message": f"Ετοιμάζω tools: {', '.join(candidate_apps) if candidate_apps else 'auto'}..."},
        )

        entity_id = f"user_{user_id}"

        try:
            session = composio.create(user_id=entity_id)

            # Candidate apps are hints from the router. Only hard-block when the user has
            # no active connections at all, or they explicitly asked for a specific toolkit.
            missing_toolkits: list[str] = []
            connected_toolkits: set[str] = set()
            try:
                toolkits = _normalize_toolkits(session.toolkits())
                connected_toolkits = {t["slug"] for t in toolkits if t.get("is_active")}
                if candidate_apps:
                    missing_toolkits = [a for a in candidate_apps if a.lower() not in connected_toolkits]
            except Exception:
                missing_toolkits = []

            msg_l = (message or "").lower()
            explicit_missing = [t for t in missing_toolkits if t.lower() in msg_l]

            if missing_toolkits and (not connected_toolkits or explicit_missing):
                links: Dict[str, str] = {}
                for toolkit in (explicit_missing or missing_toolkits):
                    try:
                        req = session.authorize(toolkit=toolkit, callback_url=f"{APP_URL}/api/oauth/callback")
                        links[toolkit] = getattr(req, "redirect_url", "") or getattr(req, "redirectUrl", "") or ""
                    except Exception:
                        links[toolkit] = ""

                msg = (
                    "Για να συνεχίσω χρειάζεται να συνδέσεις: "
                    + ", ".join(explicit_missing or missing_toolkits)
                    + ". Άνοιξε τα links και ξαναστείλε το ίδιο αίτημα."
                )

                await db.update_task(task_id, status=db.TaskStatus.FAILED, error=msg)
                publish_event(task_id, "final", {"response": msg, "needs_connection": True, "links": links})
                return

            # If router suggested apps but some aren't connected, ignore them and proceed with what is connected.
            if candidate_apps and connected_toolkits:
                candidate_apps = [a for a in candidate_apps if a.lower() in connected_toolkits]

            available_toolkits = sorted(connected_toolkits) if connected_toolkits else []
            if candidate_apps:
                available_toolkits = [a.lower() for a in candidate_apps]

            # "Agency" style: expose only a tiny tool surface and let the model search tools.
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "SEARCH_TOOLS",
                        "description": "Search available tools for the user. Use this first.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string", "description": "What you want to do, e.g. 'list my last 5 repos'"},
                                "toolkits": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Optional list of toolkit slugs to constrain search (e.g. ['github']).",
                                },
                                "limit": {"type": "integer", "description": "Max tools to return (<= 15).", "default": 8},
                            },
                            "required": ["query"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "EXECUTE_TOOL",
                        "description": "Execute a tool by slug with the provided arguments.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "slug": {"type": "string", "description": "Tool slug, e.g. GITHUB_LIST_REPOS"},
                                "arguments": {"type": ["object", "string", "null"], "description": "Tool arguments as an object. Use {} if none."},
                                "toolkit": {"type": "string", "description": "Optional toolkit slug if it cannot be inferred."},
                            },
                            "required": ["slug"],
                        },
                    },
                },
            ]

            publish_event(
                task_id,
                "tools_selected",
                {"count": len(tools), "apps": available_toolkits, "tools": ["SEARCH_TOOLS", "EXECUTE_TOOL"]},
            )

        except Exception as e:
            error_msg = f"Σφάλμα Composio: {str(e)}"
            await db.update_task(task_id, status=db.TaskStatus.FAILED, error=error_msg)
            publish_event(task_id, "final", {"response": error_msg, "needs_connection": True})
            return
        
        # ============================================
        # Step 4: Execute with tools
        # ============================================
        publish_event(task_id, "progress", {"step": "executing", "message": "Εκτελώ με tools..."})
        
        # Build messages for executor
        exec_messages = [
            {"role": "system", "content": f"""Είσαι ένας AI assistant με πρόσβαση σε εργαλεία.
Χρησιμοποίησε τα διαθέσιμα tools για να ολοκληρώσεις το αίτημα του χρήστη.
Απάντησε στα ελληνικά αν ο χρήστης γράφει ελληνικά.

ΚΡΊΣΙΜΑ ΒΉΜΑΤΑ:
1. **ΠΡΩΤΑ** κάλεσε `SEARCH_TOOLS` με query που περιγράφει τι θέλεις να κάνεις (π.χ. "list my repositories").
2. Από τα results, **ΔΙΑΛΕΞΕ ΤΟ ΑΚΡΙΒΕΣ `slug`** που ταιριάζει (π.χ. "GITHUB_LIST_REPOSITORIES_FOR_THE_AUTHENTICATED_USER")
3. **ΜΕΤΑ** κάλεσε `EXECUTE_TOOL` με το **ΠΛΗΡΕΣ, ΑΚΡΙΒΕΣ slug** που βρήκες.
   - ⚠️ **ΜΗΝ** συντομεύσεις/απλοποιήσεις το slug (π.χ. ΜΗΝ γράψεις "GITHUB_LIST_REPOS")
   - ⚠️ **ΧΡΗΣΙΜΟΠΟΙΗΣΕ ΑΚΡΙΒΩΣ** το slug από τα search results

ΠΡΟΤΕΡΑΙΟΤΗΤΑ ΑΣΦΑΛΕΙΑΣ:
- Προτίμησε read-only actions (LIST/GET/SEARCH/FETCH).
- ΜΗΝ κάνεις write actions (CREATE/UPDATE/DELETE/SEND/POST) εκτός αν ο χρήστης το ζητά ξεκάθαρα.

Available tool kits: {', '.join(candidate_apps) if candidate_apps else 'auto'}"""},
            {"role": "user", "content": message},
        ]
        
        # Call OpenAI with Composio tools
        response = client.chat.completions.create(
            model=EXECUTOR_MODEL,
            messages=exec_messages,
            tools=tools if tools else None,
            temperature=0.7,
            max_tokens=2000,
        )

        assistant_message = response.choices[0].message

        # If tools are available but the model didn't call any, nudge once to avoid empty/handwavy answers.
        if tools and not assistant_message.tool_calls:
            exec_messages.append(
                {
                    "role": "user",
                    "content": "Χρησιμοποίησε τα διαθέσιμα tools για να απαντήσεις. "
                    "Μην απαντήσεις χωρίς να κάνεις τουλάχιστον ένα tool call.",
                }
            )
            response = client.chat.completions.create(
                model=EXECUTOR_MODEL,
                messages=exec_messages,
                tools=tools,
                temperature=0.3,
                max_tokens=2000,
            )
            assistant_message = response.choices[0].message

        tool_calls_made = 0
        max_tool_calls = 6

        msg_l = (message or "").lower()
        write_hints = [
            "create",
            "update",
            "delete",
            "remove",
            "add",
            "accept",
            "invite",
            "send",
            "post",
            "merge",
            "close",
            "approve",
            "write",
            "edit",
            "σβησε",
            "διαγραψε",
            "στειλε",
            "δημιουργησε",
            "αποδεξου",
            "προσθεσε",
        ]
        allow_writes = any(h in msg_l for h in write_hints)
        write_markers = ["CREATE", "UPDATE", "DELETE", "REMOVE", "ADD_", "ACCEPT", "INVITE", "SEND", "POST", "MERGE", "CLOSE", "APPROVE", "ASSIGN", "TRANSFER", "SET_"]

        def infer_toolkit_slug(tool_slug: str) -> str:
            if not tool_slug:
                return ""
            head = tool_slug.split("_", 1)[0]
            return head.lower()

        # Build a toolkit->connected_account_id map for this user once.
        accounts_resp = composio.connected_accounts.list(user_ids=[entity_id], statuses=["ACTIVE"])
        accounts = _normalize_connected_accounts(accounts_resp)
        toolkit_to_account: Dict[str, str] = {}
        for acc in accounts:
            toolkit_to_account.setdefault(acc["toolkit_slug"].lower(), acc["id"])

        allowed_slugs: set[str] = set()
        last_search_slugs: list[str] = []
        last_recommended_slug: str | None = None
        executed_any_tool = False

        while assistant_message.tool_calls and tool_calls_made < max_tool_calls:
            tool_outputs: list[Dict[str, Any]] = []

            for tool_call in assistant_message.tool_calls:
                tool_calls_made += 1
                tool_name = tool_call.function.name
                try:
                    params = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
                except Exception:
                    params = {}

                publish_event(task_id, "tool_call", {"tool": tool_name, "params": params})

                try:
                    if tool_name == "SEARCH_TOOLS":
                        query = (params.get("query") or params.get("q") or "").strip()
                        if not query:
                            raise RuntimeError("Missing required param: query")

                        query_l = query.lower()
                        repo_mode = ("repo" in query_l) or ("repos" in query_l) or ("repository" in query_l) or ("repositories" in query_l)
                        email_mode = ("email" in query_l) or ("emails" in query_l) or ("mail" in query_l) or ("inbox" in query_l)

                        limit = params.get("limit", 8)
                        try:
                            limit = int(limit)
                        except Exception:
                            limit = 8
                        limit = max(1, min(limit, 15))

                        requested_toolkits = params.get("toolkits")
                        if requested_toolkits is None:
                            toolkits = available_toolkits or None
                        else:
                            toolkits = [str(t).lower() for t in (requested_toolkits or []) if str(t).strip()]
                            if available_toolkits:
                                toolkits = [t for t in toolkits if t in available_toolkits]
                            if not toolkits:
                                raise RuntimeError("No allowed/connected toolkits for this search.")

                        fetched = composio.tools.get(
                            user_id=entity_id,
                            toolkits=toolkits or None,
                            search=query,
                            limit=1000,
                        )

                        # Score and rank tools by relevance
                        raw_tools = []
                        print(f"🔎 SEARCH_TOOLS query='{query}' returned {len(fetched or [])} raw tools. Filtering...")
                        
                        for t in fetched or []:
                            fn = t.get("function") or {}
                            slug = str(fn.get("name") or "")
                            if not slug:
                                continue
                            
                            slug_u = slug.upper()
                            
                            # Relax write filter: if it has WRITE marker but also LIST/GET/READ/SEARCH, allow it.
                            is_write = any(m in slug_u for m in write_markers)
                            if (not allow_writes) and is_write:
                                if not any(safe in slug_u for safe in ["LIST", "GET", "READ", "SEARCH", "FETCH"]):
                                    continue
                            
                            params_schema = fn.get("parameters") or {}
                            raw_tools.append({
                                "slug": slug,
                                "slug_u": slug_u,
                                "toolkit": infer_toolkit_slug(slug),
                                "description": fn.get("description") or "",
                                "required": list(params_schema.get("required") or []),
                            })
                        
                        # Debug: Print first 10 raw tools before ranking
                        print(f"🔎 First 10 tools from Composio: {[t['slug'] for t in raw_tools[:10]]}")
                        
                        # Score tools based on query relevance
                        def score_tool(tool: Dict[str, Any]) -> int:
                            score = 0
                            slug_u = tool["slug_u"]
                            query_u = query.upper()
                            
                            # Exact keyword matches
                            if "REPOSITORIES" in query_u or "REPO" in query_u:
                                if "REPOSITORIES" in slug_u or "REPOS" in slug_u:
                                    score += 100
                                if "LIST" in slug_u:
                                    score += 50
                                if "AUTHENTICATED_USER" in slug_u or "FOR_THE_AUTHENTICATED_USER" in slug_u:
                                    score += 50
                                # Penalize unrelated repo tools
                                if "INVITATION" in slug_u or "WEBHOOK" in slug_u or "DEPLOY" in slug_u:
                                    score -= 100
                            
                            if "EMAIL" in query_u or "MESSAGE" in query_u or "INBOX" in query_u:
                                if any(k in slug_u for k in ["EMAIL", "MESSAGE", "MAIL", "INBOX"]):
                                    score += 100
                                if "LIST" in slug_u or "FETCH" in slug_u:
                                    score += 50
                            
                            # Prefer read-only actions
                            if any(safe in slug_u for safe in ["LIST", "GET", "FETCH", "SEARCH", "READ"]):
                                score += 30
                            
                            return score
                        
                        # Sort by relevance score
                        ranked_tools = sorted(raw_tools, key=score_tool, reverse=True)
                        
                        # Take top results
                        results = []
                        for tool in ranked_tools[:limit]:
                            results.append({
                                "slug": tool["slug"],
                                "toolkit": tool["toolkit"],
                                "description": tool["description"],
                                "required": tool["required"],
                            })
                        
                        print(f"🔎 SEARCH_TOOLS filtered down to {len(results)} results.")
                        if results:
                            print(f"🔎 Top 3 results: {[r['slug'] for r in results[:3]]}")
                        
                        allowed_slugs.update(r["slug"] for r in results if r.get("slug"))
                        last_search_slugs = [r["slug"] for r in results if r.get("slug")]

                        # Heuristic "best pick" to reduce model guessing.
                        last_recommended_slug = None
                        if repo_mode:
                            for r in results:
                                su = str(r.get("slug") or "").upper()
                                if ("REPO" in su or "REPOSITORY" in su) and ("LIST" in su or "GET" in su):
                                    last_recommended_slug = str(r.get("slug") or "")
                                    break
                        if email_mode and not last_recommended_slug:
                            for r in results:
                                su = str(r.get("slug") or "").upper()
                                if ("MESSAGE" in su or "EMAIL" in su or "INBOX" in su) and ("FETCH" in su or "LIST" in su or "GET" in su):
                                    last_recommended_slug = str(r.get("slug") or "")
                                    break

                        payload = {
                            "query": query,
                            "toolkits": toolkits,
                            "recommended_slug": last_recommended_slug,
                            "results": results,
                        }
                        tool_outputs.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": json.dumps(payload, ensure_ascii=False),
                            }
                        )
                        publish_event(
                            task_id,
                            "tool_result",
                            {"tool": tool_name, "success": True, "results": last_search_slugs[:10], "recommended": last_recommended_slug},
                        )

                    elif tool_name == "EXECUTE_TOOL":
                        slug = str(params.get("slug") or "").strip()
                        arguments = params.get("arguments")
                        if not slug:
                            raise RuntimeError("Missing required param: slug")
                        if arguments is None:
                            arguments = {}
                        elif isinstance(arguments, str):
                            try:
                                arguments = json.loads(arguments)
                            except Exception:
                                arguments = {}
                        if not isinstance(arguments, dict):
                            raise RuntimeError("Missing/invalid param: arguments (object)")
                        if allowed_slugs and slug not in allowed_slugs:
                            # Prefer recommended slug from the last search (avoids hallucinated slugs).
                            if last_recommended_slug and last_recommended_slug in allowed_slugs:
                                slug = last_recommended_slug
                            else:
                                raise RuntimeError(
                                    f"Tool slug '{slug}' not found in SEARCH_TOOLS results. "
                                    f"Pick one of: {last_search_slugs[:10]}"
                                )

                        if (not allow_writes) and any(m in slug.upper() for m in write_markers):
                            raise RuntimeError("Write tool blocked in MVP. Ask the user to confirm explicitly.")

                        toolkit = str(params.get("toolkit") or "").strip().lower() or infer_toolkit_slug(slug)
                        if not toolkit:
                            raise RuntimeError("Could not infer toolkit for tool slug.")
                        if available_toolkits and toolkit not in available_toolkits:
                            raise RuntimeError(f"Toolkit '{toolkit}' is not connected for this user.")

                        connected_account_id = toolkit_to_account.get(toolkit)
                        if not connected_account_id:
                            raise RuntimeError(f"No active connected account found for toolkit '{toolkit}'.")

                        print(f"🔧 EXECUTE_TOOL: slug={slug}, arguments={arguments}, entity_id={entity_id}, connected_account_id={connected_account_id}")
                        
                        try:
                            result = composio.tools.execute(
                                action=slug,  # SDK expects 'action' not 'slug'
                                params=arguments,  # SDK expects 'params' not 'arguments'
                                entity_id=entity_id,  # SDK expects 'entity_id' not 'user_id'
                                connected_account_id=connected_account_id,
                                dangerously_skip_version_check=True,
                            )
                        except Exception as exec_error:
                            print(f"❌ EXECUTE_TOOL API Error: {str(exec_error)}")
                            raise
                        if hasattr(result, "model_dump"):
                            result = result.model_dump()
                        elif hasattr(result, "dict"):
                            result = result.dict()

                        tool_outputs.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": json.dumps(result, ensure_ascii=False, default=str),
                            }
                        )
                        executed_any_tool = True
                        publish_event(task_id, "tool_result", {"tool": slug, "success": True})

                    else:
                        raise RuntimeError(f"Unknown tool: {tool_name}")

                except Exception as e:
                    tool_outputs.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps({"error": str(e)}, ensure_ascii=False),
                        }
                    )
                    publish_event(task_id, "tool_result", {"tool": tool_name, "success": False, "error": str(e)})

            exec_messages.append(assistant_message)
            exec_messages.extend(tool_outputs)

            response = client.chat.completions.create(
                model=EXECUTOR_MODEL,
                messages=exec_messages,
                tools=tools,
                temperature=0.7,
                max_tokens=2000,
            )
            assistant_message = response.choices[0].message

        if not executed_any_tool:
            raise RuntimeError("No EXECUTE_TOOL succeeded. Run SEARCH_TOOLS then EXECUTE_TOOL using the returned slug.")
        
        # Final response
        final_response = assistant_message.content or "Η εργασία ολοκληρώθηκε."
        
        await db.update_task(
            task_id,
            status=db.TaskStatus.SUCCEEDED,
            result={"response": final_response}
        )
        
        publish_event(task_id, "final", {"response": final_response})
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Task error: {error_msg}")
        
        await db.update_task(
            task_id,
            status=db.TaskStatus.FAILED,
            error=error_msg
        )
        
        publish_event(task_id, "error", {"message": error_msg})


# ============================================
# Auth Endpoints
# ============================================

@app.post("/api/auth/register")
async def register(user: UserRegister, response: Response):
    existing = await db.get_user_by_email(user.email)
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    user_id = secrets.token_hex(16)
    await db.create_user(user_id, user.email, user.name, hash_password(user.password))
    
    token = create_token(user_id)
    response.set_cookie(
        key="auth_token",
        value=token,
        httponly=True,
        max_age=60*60*24*7,
        samesite="lax"
    )
    
    return {"user": {"id": user_id, "email": user.email, "name": user.name}, "token": token}


@app.post("/api/auth/login")
async def login(credentials: UserLogin, response: Response):
    user = await db.get_user_by_email(credentials.email)
    
    if not user or user["password_hash"] != hash_password(credentials.password):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    token = create_token(user["id"])
    response.set_cookie(
        key="auth_token",
        value=token,
        httponly=True,
        max_age=60*60*24*7,
        samesite="lax"
    )
    
    return {"user": {"id": user["id"], "email": user["email"], "name": user["name"]}, "token": token}


@app.post("/api/auth/logout")
async def logout(response: Response):
    response.delete_cookie("auth_token")
    return {"message": "Logged out"}


@app.get("/api/auth/me")
async def get_me(request: Request):
    user = await get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return {"id": user["id"], "email": user["email"], "name": user["name"]}


# ============================================
# Chat Endpoints
# ============================================

@app.post("/v1/chat/send", response_model=ChatSendResponse)
async def chat_send(request: Request, chat_request: ChatSendRequest):
    """Send a chat message"""
    user = await get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    task_id = await db.create_task(user["id"])
    task_events[task_id] = []
    
    conversation_history = [{"role": m.role, "content": m.content} for m in chat_request.conversation_history]
    asyncio.create_task(process_task_with_composio(task_id, user["id"], chat_request.message, conversation_history))
    
    return ChatSendResponse(task_id=task_id, status="queued")


@app.get("/v1/chat/stream")
async def chat_stream(request: Request, task_id: str):
    """SSE stream for task progress"""
    user = await get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    task = await db.get_task(task_id)
    if not task or task["user_id"] != user["id"]:
        raise HTTPException(status_code=404, detail="Task not found")
    
    async def event_generator():
        last_index = 0
        
        yield {"event": "status", "data": json.dumps({"status": task["status"]})}
        
        if task["status"] in ["succeeded", "failed", "expired"]:
            if task.get("result"):
                yield {"event": "final", "data": json.dumps(task["result"])}
            return
        
        max_wait = 120
        waited = 0
        
        while waited < max_wait:
            if await request.is_disconnected():
                break
            
            events = task_events.get(task_id, [])
            
            while last_index < len(events):
                event = events[last_index]
                yield {"event": event["type"], "data": json.dumps(event["data"], ensure_ascii=False)}
                
                if event["type"] in ["final", "error", "expired"]:
                    return
                
                last_index += 1
            
            await asyncio.sleep(0.1)
            waited += 0.1
    
    return EventSourceResponse(event_generator())


@app.get("/v1/tasks/{task_id}")
async def get_task(request: Request, task_id: str):
    """Get task status"""
    user = await get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    task = await db.get_task(task_id)
    if not task or task["user_id"] != user["id"]:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return {
        "id": task["id"],
        "status": task["status"],
        "result": task.get("result"),
        "error": task.get("error"),
        "router_output": task.get("router_output")
    }


# ============================================
# Connections Endpoints (Composio)
# ============================================

@app.get("/api/connections")
async def get_connections(request: Request):
    """Get user's connected apps via Composio"""
    user = await get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    entity_id = f"user_{user['id']}"
    
    try:
        session = composio.create(user_id=entity_id)
        toolkits = _normalize_toolkits(session.toolkits())
        connected = [t for t in toolkits if t.get("is_active")]
        return {"connections": connected, "all": toolkits}
    except Exception as e:
        return {"connections": [], "error": str(e)}


@app.post("/api/connections/{app_name}/connect")
async def connect_app(app_name: str, request: Request):
    """Get OAuth URL to connect an app via Composio"""
    user = await get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    entity_id = f"user_{user['id']}"
    
    try:
        session = composio.create(user_id=entity_id)
        # Get auth request using the correct authorize() method
        connection_request = session.authorize(
            toolkit=app_name,
            callback_url=f"{APP_URL}/api/oauth/callback"
        )
        # ConnectionRequest has redirect_url property
        return {"redirect_url": connection_request.redirect_url}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/oauth/callback")
async def oauth_callback():
    return RedirectResponse(url="/?connected=true")


@app.get("/api/apps")
async def list_apps():
    """List available apps from Composio"""
    try:
        # Get available toolkits
        toolkits = composio.get_toolkits()
        popular = ["gmail", "googlecalendar", "outlook", "github", "slack", "notion", "twitter", "linear"]
        filtered = [t for t in toolkits if t.get("key", "").lower() in popular]
        return {"apps": filtered if filtered else toolkits[:50]}
    except Exception as e:
        # Fallback to hardcoded list
        return {"apps": [
            {"key": "gmail", "name": "Gmail", "description": "Send and read emails"},
            {"key": "outlook", "name": "Outlook", "description": "Send and read emails via Microsoft Outlook"},
            {"key": "github", "name": "GitHub", "description": "Manage repos and issues"},
            {"key": "slack", "name": "Slack", "description": "Send messages to channels"},
            {"key": "notion", "name": "Notion", "description": "Create pages and databases"},
            {"key": "googlecalendar", "name": "Google Calendar", "description": "Manage calendar events"}
        ]}


# ============================================
# Health
# ============================================

@app.get("/api/health")
async def health():
    return {"status": "healthy", "version": "3.2.0", "mode": "composio-sdk"}


# ============================================
# Static Files & Root
# ============================================

@app.get("/")
async def root():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "AI Agent MVP API", "docs": "/docs"}


if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
