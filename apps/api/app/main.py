from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.core.config import settings
from app.routers import auth, chat, health, ingest, memory_router, person, persona_router, telegram, voice
from app.services.telegram_bot import start_telegram_polling, stop_telegram_polling


@asynccontextmanager
async def lifespan(app: FastAPI):
    auth_mode = "on" if settings.family_access_key or settings.admin_access_key else "OFF (dev)"
    logger.info("Starting Memorial Assistant API (env={}, auth={})", settings.app_env, auth_mode)
    start_telegram_polling()
    yield
    stop_telegram_polling()
    logger.info("Shutting down Memorial Assistant API")


app = FastAPI(
    title="Memorial Voice & Chat Assistant",
    version="0.2.0",
    lifespan=lifespan,
    docs_url=None if settings.is_production else "/docs",
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()],
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "X-Access-Key"],
)

for module in (health, auth, chat, voice, telegram, ingest, persona_router, person, memory_router):
    app.include_router(module.router)
