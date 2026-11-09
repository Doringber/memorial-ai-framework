"""Telegram Bot router — webhook receiver and admin webhook registration endpoint."""

from __future__ import annotations

import secrets

from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.security import require_admin
from app.services.telegram_bot import handle_telegram_update, register_telegram_webhook

router = APIRouter(prefix="/api/telegram", tags=["telegram"])


class RegisterWebhookRequest(BaseModel):
    public_url: str = Field(min_length=8, max_length=500, description="Public HTTPS base URL, e.g. https://ronny.example.com")


class RegisterWebhookResponse(BaseModel):
    status: str
    webhook_url: str
    bot_username: str
    chat_link: str


def _verify_webhook_secret(secret_header: str) -> None:
    expected = settings.telegram_webhook_secret.strip()
    if not expected:
        return
    if not secrets.compare_digest(secret_header.encode(), expected.encode()):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid Telegram webhook secret")


@router.post("/webhook")
def telegram_webhook(
    update: dict,
    x_telegram_bot_api_secret_token: str = Header(default=""),
) -> dict:
    """Receive incoming Telegram Bot updates (text and voice messages)."""
    if not settings.telegram_bot_token:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "TELEGRAM_BOT_TOKEN is not configured")
    _verify_webhook_secret(x_telegram_bot_api_secret_token)
    return handle_telegram_update(update)


@router.post("/register-webhook", response_model=RegisterWebhookResponse, dependencies=[Depends(require_admin)])
def register_webhook(req: RegisterWebhookRequest) -> RegisterWebhookResponse:
    """Register this server's HTTPS URL as the Telegram Bot webhook and return the t.me chat link."""
    if not settings.telegram_bot_token:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "TELEGRAM_BOT_TOKEN is not configured")
    try:
        result = register_telegram_webhook(req.public_url)
    except ValueError as e:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(e)) from e
    return RegisterWebhookResponse(**result)
