"""
Configuration for AI Agent MVP
"""
import os
import secrets
from dotenv import load_dotenv

load_dotenv()

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ROUTER_MODEL = os.getenv("ROUTER_MODEL", "gpt-4o-mini")  # Cheap model for routing
EXECUTOR_MODEL = os.getenv("EXECUTOR_MODEL", "gpt-4o-mini")

# Composio
COMPOSIO_API_KEY = os.getenv("COMPOSIO_API_KEY")
COMPOSIO_BASE_URL = os.getenv("COMPOSIO_BASE_URL", "https://backend.composio.dev/api")

# Auth
JWT_SECRET = os.getenv("JWT_SECRET", secrets.token_hex(32))
APP_URL = os.getenv("APP_URL", "http://localhost:8000")

# Redis
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Limits & Timeouts
MAX_TOOL_CALLS_PER_TASK = 3
MAX_RETRIES = 2
CONFIRM_TTL_MINUTES = 30
CONFIRM_TTL_SEND_MINUTES = 15

# Prompt versions (for A/B testing / debugging)
ROUTER_PROMPT_VERSION = "router_v1.0.0"
EXECUTOR_PROMPT_VERSION = "executor_v1.0.0"
