"""API Dependencies for GEAK Online Service."""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Annotated

from server.services.auth import AuthService
from server.services.task_manager import TaskManager

# Security scheme
security = HTTPBearer()


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)]
) -> dict:
    """Get current authenticated user.
    
    Validates the API key against SaFE platform and returns user info.
    """
    api_key = credentials.credentials
    auth = AuthService(api_key)
    user_info = await auth.get_user_info()
    # Add api_key to user_info for downstream use
    user_info["_api_key"] = api_key
    return user_info


async def get_task_manager(
    user: Annotated[dict, Depends(get_current_user)]
) -> TaskManager:
    """Get task manager for current user."""
    user_id = user.get("id") or user.get("userId")
    api_key = user.get("_api_key")
    return TaskManager(user_id=user_id, api_key=api_key)


# Type aliases for cleaner route signatures
CurrentUser = Annotated[dict, Depends(get_current_user)]
CurrentTaskManager = Annotated[TaskManager, Depends(get_task_manager)]
