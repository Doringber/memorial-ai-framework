"""Talking-head video: reply audio → lip-synced mp4 of the person's portrait.

Rendering runs in a separate local process (apps/avatar-sidecar) because the
model stack (AvatarForcing on MLX + PyTorch MPS, transformers 4.30) can't share
this API's environment. That process loads the model and the portrait once;
this module only sends it the TTS audio and stores the mp4 it returns.

Same rules as tts.py: a failed render is never retried and never falls back,
the reply simply stays text (or voice) only.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import httpx
from loguru import logger

from app.core.config import is_local_service_url, settings

VideoUnavailableReason = Literal["off", "blocked", "down", "failed"]


class VideoUnavailableError(RuntimeError):
    """Video is off, not allowed, the sidecar is unreachable, or the render failed."""

    def __init__(self, reason: VideoUnavailableReason, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason


@dataclass(frozen=True)
class VideoInfo:
    enabled: bool
    available: bool


def _service_allowed() -> bool:
    return is_local_service_url(settings.avatar_service_url) or settings.allow_external_services


def _base_url() -> str:
    return settings.avatar_service_url.rstrip("/")


def video_info() -> VideoInfo:
    """Whether video is switched on, and whether the sidecar answers its health check."""
    if not settings.avatar_video_enabled or not _service_allowed():
        return VideoInfo(enabled=settings.avatar_video_enabled, available=False)
    try:
        with httpx.Client(timeout=3.0) as client:
            ok = client.get(f"{_base_url()}/health").status_code == 200
    except httpx.HTTPError:
        ok = False
    return VideoInfo(enabled=True, available=ok)


def _prune_old_video() -> None:
    cutoff = time.time() - settings.audio_retention_hours * 3600
    for path in settings.video_output_dir.glob("*.mp4"):
        if path.stat().st_mtime < cutoff:
            path.unlink(missing_ok=True)


def render_avatar_video(audio_path: Path) -> Path:
    """Render `audio_path` (the TTS mp3) as a talking-head mp4. Returns the mp4 path."""
    if not settings.avatar_video_enabled:
        raise VideoUnavailableError("off", "AVATAR_VIDEO_ENABLED is false")
    if not _service_allowed():
        raise VideoUnavailableError("blocked", "AVATAR_SERVICE_URL is not on this machine")
    if not audio_path.is_file():
        raise ValueError(f"No audio to render: {audio_path.name}")

    settings.video_output_dir.mkdir(parents=True, exist_ok=True)
    _prune_old_video()
    output_file = settings.video_output_dir / f"{uuid.uuid4()}.mp4"

    started = time.perf_counter()
    try:
        with httpx.Client(timeout=float(settings.avatar_video_timeout_s)) as client:
            res = client.post(
                f"{_base_url()}/render",
                content=audio_path.read_bytes(),
                headers={
                    "Content-Type": "application/octet-stream",
                    "X-Max-Seconds": str(settings.avatar_video_max_seconds),
                },
            )
    except httpx.ConnectError as e:
        raise VideoUnavailableError("down", "Avatar sidecar is not running") from e
    except httpx.HTTPError as e:
        logger.error("Avatar render request failed ({})", type(e).__name__)
        raise VideoUnavailableError("failed", "Avatar render request failed") from e

    if res.status_code != 200 or not res.content:
        logger.error("Avatar render failed (HTTP {})", res.status_code)
        raise VideoUnavailableError("failed", f"Avatar sidecar returned HTTP {res.status_code}")

    output_file.write_bytes(res.content)
    output_file.chmod(0o600)
    logger.info(
        "Avatar video rendered: {} ({} bytes, {:.1f}s)",
        output_file.name,
        output_file.stat().st_size,
        time.perf_counter() - started,
    )
    return output_file
