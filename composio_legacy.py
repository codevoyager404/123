"""
Composio Client for executing actions
"""
import httpx
from typing import Dict, Any
from config import COMPOSIO_API_KEY, COMPOSIO_BASE_URL


async def execute_composio_action(action_name: str, entity_id: str, params: Dict) -> Dict[str, Any]:
    """Execute a Composio action for a specific entity"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(
                f"{COMPOSIO_BASE_URL}/v2/actions/{action_name}/execute",
                headers={
                    "x-api-key": COMPOSIO_API_KEY,
                    "Content-Type": "application/json"
                },
                json={
                    "entityId": entity_id,
                    "input": params
                }
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            return {"error": f"HTTP_{e.response.status_code}", "details": str(e)}
        except httpx.TimeoutException:
            return {"error": "timeout", "details": "Request timed out"}
        except Exception as e:
            return {"error": "connection_error", "details": str(e)}


async def get_available_apps() -> list:
    """Get list of available apps from Composio"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(
                f"{COMPOSIO_BASE_URL}/v1/apps",
                headers={"x-api-key": COMPOSIO_API_KEY},
                params={"limit": 50}
            )
            data = response.json()
            return data.get("items", [])
        except Exception as e:
            print(f"Error getting apps: {e}")
            return []


async def get_connection_url(entity_id: str, app_name: str, redirect_uri: str) -> str:
    """Get OAuth URL for connecting an app"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(
                f"{COMPOSIO_BASE_URL}/v1/connectedAccounts",
                headers={"x-api-key": COMPOSIO_API_KEY},
                json={
                    "integrationId": app_name.lower(),
                    "entityId": entity_id,
                    "redirectUri": redirect_uri
                }
            )
            data = response.json()
            return data.get("redirectUrl", "")
        except Exception as e:
            print(f"Error getting connection URL: {e}")
            return ""


async def get_entity_connections(entity_id: str) -> list:
    """Get all connections for an entity"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(
                f"{COMPOSIO_BASE_URL}/v1/connectedAccounts",
                headers={"x-api-key": COMPOSIO_API_KEY},
                params={"entityId": entity_id}
            )
            data = response.json()
            return data.get("items", [])
        except Exception as e:
            print(f"Error getting connections: {e}")
            return []
