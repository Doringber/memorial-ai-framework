"""Shared-password access control.

Two roles: "family" can chat and save facts; "admin" can also ingest files,
add memories directly, and manage persons/personas. The admin key is
accepted wherever the family key is.
"""

from __future__ import annotations

import secrets
from typing import Literal

from fastapi import Depends, Header, HTTPException, status

from app.core.config import settings

Role = Literal["family", "admin"]


def _matches(provided: str, expected: str) -> bool:
    return bool(expected) and secrets.compare_digest(provided.encode(), expected.encode())


def resolve_role(access_key: str) -> Role | None:
    """Return the role granted by `access_key`, or None if it grants nothing."""
    if _matches(access_key, settings.admin_access_key):
        return "admin"
    if _matches(access_key, settings.family_access_key):
        return "family"
    return None


def _auth_disabled() -> bool:
    return not settings.is_production and not (
        settings.family_access_key or settings.admin_access_key
    )


def require_family(x_access_key: str = Header(default="")) -> Role:
    if _auth_disabled():
        return "admin"
    role = resolve_role(x_access_key)
    if role is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or missing access key")
    return role


def require_admin(role: Role = Depends(require_family)) -> Role:
    if role != "admin":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Admin access required")
    return role
