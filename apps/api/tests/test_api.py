import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings, settings
from app.main import app
from tests.conftest import PERSON_ID, FakeLLM

FAMILY = {"X-Access-Key": "family-pass"}
ADMIN = {"X-Access-Key": "admin-pass"}


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(settings, "family_access_key", "family-pass")
    monkeypatch.setattr(settings, "admin_access_key", "admin-pass")
    return TestClient(app)


def test_health_is_public(client: TestClient) -> None:
    assert client.get("/health").json()["status"] == "ok"


@pytest.mark.parametrize(("headers", "role"), [(FAMILY, "family"), (ADMIN, "admin")])
def test_session_reports_role(client: TestClient, headers: dict, role: str) -> None:
    assert client.get("/api/auth/session", headers=headers).json() == {"role": role}


@pytest.mark.parametrize("headers", [{}, {"X-Access-Key": "wrong"}])
def test_chat_requires_access_key(client: TestClient, headers: dict) -> None:
    res = client.post("/api/chat/text", json={"person_id": PERSON_ID, "message": "היי"}, headers=headers)
    assert res.status_code == 401


def test_speak_reports_voice_off(client: TestClient) -> None:
    res = client.post("/api/chat/speak", json={"text": "שלום"}, headers=FAMILY)
    assert res.status_code == 503
    assert res.json()["detail"] == "off"


def test_voice_info(client: TestClient) -> None:
    assert client.get("/api/chat/voice-info").status_code == 401
    info = client.get("/api/chat/voice-info", headers=FAMILY).json()
    assert info == {"provider": "none", "external": False, "available": False, "chars_used": 0, "chars_limit": 0}


def test_family_can_chat(client: TestClient, fake_llm: FakeLLM) -> None:
    res = client.post("/api/chat/text", json={"person_id": PERSON_ID, "message": "מה נשמע?"}, headers=FAMILY)
    assert res.status_code == 200
    assert res.json()["reply"] == fake_llm.reply


def test_chat_stream_and_spoken_commit_endpoints(client: TestClient, fake_llm: FakeLLM) -> None:
    fake_llm.reply = "משפט ראשון לבדיקה.... 🥰 משפט שני שלא הושמע."
    res = client.post(
        "/api/chat/stream",
        json={"person_id": PERSON_ID, "message": "מה נשמע?", "defer_history": True},
        headers=FAMILY,
    )
    assert res.status_code == 200
    events = [
        __import__("json").loads(line)
        for line in res.text.splitlines()
        if line.strip()
    ]
    assert [e["type"] for e in events] == ["meta", "sentence", "sentence", "done"]
    sid = events[0]["session_id"]

    commit = client.post(
        "/api/chat/spoken",
        json={"session_id": sid, "user_message": "מה נשמע?", "spoken_text": events[1]["text"]},
        headers=FAMILY,
    )
    assert commit.status_code == 200
    assert commit.json()["committed_chars"] == len(events[1]["text"])

    lat = client.post(
        "/api/chat/latency",
        json={"mode": "voice", "client_e2e_ms": 1500.0, "interrupted": True, "spoken_sentences": 1},
        headers=FAMILY,
    )
    assert lat.status_code == 200


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        ("post", "/api/memory/add", {"person_id": PERSON_ID, "text": "x"}),
        ("post", "/api/persona/draft", {"person_id": PERSON_ID}),
        ("post", "/api/person/create", {"display_name": "x"}),
        ("delete", f"/api/person/{PERSON_ID}/data", None),
        ("post", f"/api/person/{PERSON_ID}/avatar", None),
    ],
)
def test_admin_endpoints_reject_family(client: TestClient, method: str, path: str, body: dict | None) -> None:
    res = client.request(method, path, json=body, headers=FAMILY)
    assert res.status_code == 403


def test_admin_can_add_memory(client: TestClient) -> None:
    res = client.post("/api/memory/add", json={"person_id": PERSON_ID, "text": "דוד אהב חמוצים"}, headers=ADMIN)
    assert res.status_code == 200
    assert res.json()["status"] == "saved"


def test_family_can_list_persons(client: TestClient) -> None:
    assert client.get("/api/person/list", headers=FAMILY).json() == []


@pytest.mark.parametrize("name", ["..%2F..%2F.env", "not-a-uuid.mp3", "00000000-0000-0000-0000-000000000000.mp3"])
def test_audio_rejects_bad_or_missing_files(client: TestClient, name: str) -> None:
    assert client.get(f"/api/chat/audio/{name}").status_code == 404


def test_document_upload_rejects_unknown_type(client: TestClient) -> None:
    res = client.post(
        "/api/ingest/document",
        data={"person_id": PERSON_ID},
        files={"file": ("x.exe", b"MZ", "application/octet-stream")},
        headers=ADMIN,
    )
    assert res.status_code == 415


JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 64
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64


def _upload_avatar(client: TestClient, name: str, content: bytes, person_id: str = PERSON_ID) -> int:
    files = {"file": (name, content, "application/octet-stream")}
    return client.post(f"/api/person/{person_id}/avatar", files=files, headers=ADMIN).status_code


def test_avatar_upload_replace_and_serve(client: TestClient) -> None:
    assert client.get(f"/api/person/{PERSON_ID}/avatar", headers=FAMILY).status_code == 404

    assert _upload_avatar(client, "roni.jpg", JPEG) == 200
    assert _upload_avatar(client, "roni.png", PNG) == 200

    res = client.get(f"/api/person/{PERSON_ID}/avatar", headers=FAMILY)
    assert res.status_code == 200
    assert res.headers["content-type"] == "image/png"
    assert res.content == PNG
    assert [p.name for p in settings.avatars_dir.iterdir()] == [f"{PERSON_ID}.png"]


def test_avatar_requires_family_key(client: TestClient) -> None:
    assert client.get(f"/api/person/{PERSON_ID}/avatar").status_code == 401


@pytest.mark.parametrize(
    ("name", "content", "person_id", "expected"),
    [
        ("roni.jpg", b"<html>not an image</html>", PERSON_ID, 415),
        ("roni.gif", b"GIF89a", PERSON_ID, 415),
        ("roni.jpg", JPEG, "..%2F..%2Fsecrets", 404),
    ],
)
def test_avatar_upload_rejects_bad_input(
    client: TestClient, name: str, content: bytes, person_id: str, expected: int
) -> None:
    assert _upload_avatar(client, name, content, person_id) == expected
    assert not settings.avatars_dir.exists() or not any(settings.avatars_dir.iterdir())


def test_production_requires_access_keys() -> None:
    with pytest.raises(ValueError, match="FAMILY_ACCESS_KEY"):
        Settings(_env_file=None, app_env="production", family_access_key="", admin_access_key="")


def test_whatsapp_onboarding_and_persona_activation(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    wa_chat = (
        "[01.01.24, 10:00:00] נועה: אבא מתי נפגשים בשבת?\n"
        "[01.01.24, 10:01:00] דוד כהן: תלוי מתי תגיעו.... הכי טוב להיות ביחד\n"
        "[01.01.24, 10:05:00] דוד כהן: סעו בזהירות ותעדכנו כשתגיעו\n"
    ).encode("utf-8")

    senders_res = client.post(
        "/api/ingest/whatsapp-senders",
        files={"file": ("_chat.txt", wa_chat, "text/plain")},
        headers=ADMIN,
    )
    assert senders_res.status_code == 200
    assert senders_res.json() == [
        {"sender": "דוד כהן", "messages": 2},
        {"sender": "נועה", "messages": 1},
    ]

    ingest_res = client.post(
        "/api/ingest/whatsapp",
        data={"person_id": PERSON_ID, "sender": "דוד כהן"},
        files={"file": ("_chat.txt", wa_chat, "text/plain")},
        headers=ADMIN,
    )
    assert ingest_res.status_code == 200
    assert ingest_res.json() == {
        "status": "ok",
        "sender": "דוד כהן",
        "style_examples": 2,
        "conversation_windows": 2,
    }

    monkeypatch.setattr(
        "app.services.persona.generate_response",
        lambda msgs: '{"name": "דוד כהן", "overall_impression": "איש משפחה חם"}',
    )
    act_res = client.post(
        "/api/persona/draft",
        json={"person_id": PERSON_ID, "display_name": "סבא דוד", "activate": True},
        headers=ADMIN,
    )
    assert act_res.status_code == 200
    assert act_res.json()["status"] == "ok"
    assert act_res.json()["persona"]["display_name"] == "סבא דוד"
    assert (settings.personas_dir / f"{PERSON_ID}.json").is_file()
