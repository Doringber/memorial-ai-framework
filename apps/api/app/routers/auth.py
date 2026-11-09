from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.core.security import Role, require_family

router = APIRouter(prefix="/api/auth", tags=["auth"])


class SessionInfo(BaseModel):
    role: Role


@router.get("/session", response_model=SessionInfo)
def session_info(role: Role = Depends(require_family)) -> SessionInfo:
    """Validate an access key and report its role (used by the web login)."""
    return SessionInfo(role=role)
