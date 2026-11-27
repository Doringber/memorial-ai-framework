"""Avatar video service: local-only guard and sidecar client behaviour (no model needed)."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from app.core.config import Settings, is_local_service_url, settings
from app.services import avatar_video
from app.services.avatar_video import VideoUnavailableError, render_avatar_video


@pytest.mark.parametrize(
    ("url", "local"),
    [
        ("http://127.0.0.1:8090", True),
        ("http://localhost:8090/", True),
        ("http://host.docker.internal:8090", True),
        ("http://[::1]:8090", True),
        ("http://10.0.0.5:8090", False),
        ("https://avatar.example.com", False),
        ("", False),
    ],
)
def test_is_local_service_url(url: str, local: bool) -> None:
    assert is_local_service_url(url) is local


def test_remote_avatar_service_is_blocked_by_default() -> None:
    with pytest.raises(ValueError, match="AVATAR_SERVICE_URL"):
        Settings(_env_file=None, avatar_video_enabled=True, avatar_service_url="https://avatar.example.com")
    # Disabled video never sends anything, so the URL doesn't matter.
    Settings(_env_file=None, avatar_video_enabled=False, avatar_service_url="https://avatar.example.com")
    Settings(
        _env_file=None,
        avatar_video_enabled=True,
        avatar_service_url="https://avatar.example.com",
        allow_external_services=True,
    )


@pytest.fixture
def mp3(tmp_path: Path) -> Path:
    path = tmp_path / "reply.mp3"
    path.write_bytes(b"ID3-fake-mp3")
    return path


def _mock_sidecar(monkeypatch: pytest.MonkeyPatch, handler) -> list[httpx.Request]:
    seen: list[httpx.Request] = []

    def wrapped(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    real_client = httpx.Client
    monkeypatch.setattr(
        avatar_video.httpx,
        "Client",
        lambda **kw: real_client(transport=httpx.MockTransport(wrapped), **kw),
    )
    return seen


def test_render_posts_audio_and_saves_mp4(mp3: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "avatar_video_enabled", True)
    monkeypatch.setattr(settings, "avatar_video_max_seconds", 30)
    seen = _mock_sidecar(monkeypatch, lambda req: httpx.Response(200, content=b"fake-mp4"))

    out = render_avatar_video(mp3)

    assert out.parent == settings.video_output_dir and out.suffix == ".mp4"
    assert out.read_bytes() == b"fake-mp4"
    assert seen[0].url == "http://127.0.0.1:8090/render"
    assert seen[0].content == b"ID3-fake-mp3"
    assert seen[0].headers["X-Max-Seconds"] == "30"


def test_render_refuses_when_disabled(mp3: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "avatar_video_enabled", False)
    seen = _mock_sidecar(monkeypatch, lambda req: httpx.Response(200, content=b"x"))
    with pytest.raises(VideoUnavailableError) as exc:
        render_avatar_video(mp3)
    assert exc.value.reason == "off"
    assert seen == []


def test_render_refuses_remote_sidecar_at_runtime(mp3: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "avatar_video_enabled", True)
    monkeypatch.setattr(settings, "avatar_service_url", "https://avatar.example.com")
    seen = _mock_sidecar(monkeypatch, lambda req: httpx.Response(200, content=b"x"))
    with pytest.raises(VideoUnavailableError) as exc:
        render_avatar_video(mp3)
    assert exc.value.reason == "blocked"
    assert seen == []


def test_render_reports_sidecar_down(mp3: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "avatar_video_enabled", True)

    def refuse(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=req)

    _mock_sidecar(monkeypatch, refuse)
    with pytest.raises(VideoUnavailableError) as exc:
        render_avatar_video(mp3)
    assert exc.value.reason == "down"
    assert list(settings.video_output_dir.glob("*.mp4")) == []


def test_render_reports_sidecar_error(mp3: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "avatar_video_enabled", True)
    _mock_sidecar(monkeypatch, lambda req: httpx.Response(500, content=b"render failed"))
    with pytest.raises(VideoUnavailableError) as exc:
        render_avatar_video(mp3)
    assert exc.value.reason == "failed"
    assert list(settings.video_output_dir.glob("*.mp4")) == []
