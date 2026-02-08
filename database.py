"""
Database Models and Operations (SQLite for MVP)
"""
import aiosqlite
import json
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from enum import Enum


DB_PATH = "agent.db"


class TaskStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    AWAITING_CONFIRM = "awaiting_confirm"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    EXPIRED = "expired"


async def init_db():
    """Initialize database schema"""
    async with aiosqlite.connect(DB_PATH) as db:
        # Users table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        
        # Conversations table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                title TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        
        # Messages table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            )
        """)
        
        # Tasks table (core of the orchestration)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                conversation_id TEXT,
                status TEXT NOT NULL DEFAULT 'queued',
                router_output TEXT,
                router_prompt_version TEXT,
                executor_prompt_version TEXT,
                confirm_expires_at TEXT,
                confirm_token TEXT,
                result TEXT,
                error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        
        # Task Steps table (individual tool calls)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS task_steps (
                id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                step_number INTEGER NOT NULL,
                tool_name TEXT,
                tool_params TEXT,
                tool_result TEXT,
                attempt INTEGER DEFAULT 0,
                idempotency_key TEXT,
                retryable INTEGER DEFAULT 1,
                last_error_code TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                created_at TEXT NOT NULL,
                FOREIGN KEY (task_id) REFERENCES tasks(id)
            )
        """)
        
        # Tool Actions catalog
        await db.execute("""
            CREATE TABLE IF NOT EXISTS tool_actions (
                id TEXT PRIMARY KEY,
                app_name TEXT NOT NULL,
                action_name TEXT NOT NULL,
                display_name TEXT,
                description TEXT,
                parameters TEXT,
                risk_level TEXT DEFAULT 'low',
                requires_confirmation INTEGER DEFAULT 0,
                tags TEXT,
                updated_at TEXT NOT NULL
            )
        """)
        
        # User connections (which apps user has connected)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS user_connections (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                app_name TEXT NOT NULL,
                connection_id TEXT,
                status TEXT DEFAULT 'active',
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id),
                UNIQUE(user_id, app_name)
            )
        """)
        
        await db.commit()


# ============================================
# User Operations
# ============================================

async def get_user_by_email(email: str) -> Optional[Dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM users WHERE email = ?", (email,)) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None


async def get_user_by_id(user_id: str) -> Optional[Dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM users WHERE id = ?", (user_id,)) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None


async def create_user(user_id: str, email: str, name: str, password_hash: str) -> Dict:
    async with aiosqlite.connect(DB_PATH) as db:
        now = datetime.utcnow().isoformat()
        await db.execute(
            "INSERT INTO users (id, email, name, password_hash, created_at) VALUES (?, ?, ?, ?, ?)",
            (user_id, email, name, password_hash, now)
        )
        await db.commit()
    return {"id": user_id, "email": email, "name": name}


# ============================================
# Task Operations
# ============================================

async def create_task(user_id: str, conversation_id: Optional[str] = None) -> str:
    """Create a new task and return its ID"""
    task_id = secrets.token_hex(16)
    now = datetime.utcnow().isoformat()
    
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            INSERT INTO tasks (id, user_id, conversation_id, status, created_at, updated_at)
            VALUES (?, ?, ?, 'queued', ?, ?)
        """, (task_id, user_id, conversation_id, now, now))
        await db.commit()
    
    return task_id


async def get_task(task_id: str) -> Optional[Dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)) as cursor:
            row = await cursor.fetchone()
            if row:
                data = dict(row)
                if data.get("router_output"):
                    data["router_output"] = json.loads(data["router_output"])
                if data.get("result"):
                    data["result"] = json.loads(data["result"])
                return data
            return None


async def update_task(task_id: str, **kwargs) -> None:
    """Update task fields"""
    if not kwargs:
        return
    
    # Serialize JSON fields
    if "router_output" in kwargs and kwargs["router_output"]:
        kwargs["router_output"] = json.dumps(kwargs["router_output"])
    if "result" in kwargs and kwargs["result"]:
        kwargs["result"] = json.dumps(kwargs["result"])
    
    kwargs["updated_at"] = datetime.utcnow().isoformat()
    
    set_clause = ", ".join(f"{k} = ?" for k in kwargs.keys())
    values = list(kwargs.values()) + [task_id]
    
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(f"UPDATE tasks SET {set_clause} WHERE id = ?", values)
        await db.commit()


async def set_task_awaiting_confirm(task_id: str, ttl_minutes: int = 30) -> str:
    """Set task to awaiting confirmation with TTL"""
    confirm_token = secrets.token_hex(16)
    expires_at = (datetime.utcnow() + timedelta(minutes=ttl_minutes)).isoformat()
    
    await update_task(
        task_id,
        status=TaskStatus.AWAITING_CONFIRM,
        confirm_token=confirm_token,
        confirm_expires_at=expires_at
    )
    
    return confirm_token


async def get_expired_tasks() -> List[Dict]:
    """Get tasks that have expired confirmation"""
    now = datetime.utcnow().isoformat()
    
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("""
            SELECT * FROM tasks 
            WHERE status = 'awaiting_confirm' 
            AND confirm_expires_at < ?
        """, (now,)) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]


# ============================================
# Task Steps Operations
# ============================================

async def create_task_step(
    task_id: str, 
    step_number: int, 
    tool_name: str, 
    tool_params: Dict,
    idempotency_key: str
) -> str:
    """Create a task step"""
    step_id = secrets.token_hex(16)
    now = datetime.utcnow().isoformat()
    
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            INSERT INTO task_steps 
            (id, task_id, step_number, tool_name, tool_params, idempotency_key, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (step_id, task_id, step_number, tool_name, json.dumps(tool_params), idempotency_key, now))
        await db.commit()
    
    return step_id


async def update_task_step(step_id: str, **kwargs) -> None:
    """Update task step"""
    if "tool_result" in kwargs and kwargs["tool_result"]:
        kwargs["tool_result"] = json.dumps(kwargs["tool_result"])
    if "tool_params" in kwargs and kwargs["tool_params"]:
        kwargs["tool_params"] = json.dumps(kwargs["tool_params"])
    
    set_clause = ", ".join(f"{k} = ?" for k in kwargs.keys())
    values = list(kwargs.values()) + [step_id]
    
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(f"UPDATE task_steps SET {set_clause} WHERE id = ?", values)
        await db.commit()


# ============================================
# User Connections
# ============================================

async def get_user_connections(user_id: str) -> List[str]:
    """Get list of app names user has connected"""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT app_name FROM user_connections WHERE user_id = ? AND status = 'active'",
            (user_id,)
        ) as cursor:
            rows = await cursor.fetchall()
            return [row["app_name"] for row in rows]


async def save_user_connection(user_id: str, app_name: str, connection_id: str) -> None:
    """Save or update a user connection"""
    now = datetime.utcnow().isoformat()
    
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            INSERT INTO user_connections (id, user_id, app_name, connection_id, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(user_id, app_name) DO UPDATE SET 
                connection_id = excluded.connection_id,
                status = 'active'
        """, (secrets.token_hex(16), user_id, app_name, connection_id, now))
        await db.commit()
