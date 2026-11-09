"""Shared fixtures: isolated data dir, real Chroma in tmp, fake embeddings, fake LLM."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import pytest

# Must run before app.core.config is imported, so the real ~/MemorialData/.env is never read.
os.environ["MEMORIAL_DATA_HOME"] = tempfile.mkdtemp(prefix="askronny-test-")

from app.core.config import settings  # noqa: E402
from app.services import chat as chat_service  # noqa: E402
from app.services import embeddings, vector_store  # noqa: E402
from app.services import persona as persona_service  # noqa: E402

PERSON_ID = "11111111-2222-3333-4444-555555555555"
_DIM = 512


def _fake_embed(texts: list[str]) -> list[list[float]]:
    """Deterministic bag-of-words hashing — similar texts share dimensions."""
    vectors = []
    for text in texts:
        vec = [0.0] * _DIM
        for word in text.lower().split():
            vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % _DIM] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        vectors.append([v / norm for v in vec])
    return vectors


@pytest.fixture(autouse=True)
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Point all storage at tmp_path, disable auth/reranker, stub embeddings."""
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    monkeypatch.setattr(settings, "chroma_persist_dir", tmp_path / "chroma")
    monkeypatch.setattr(settings, "family_access_key", "")
    monkeypatch.setattr(settings, "admin_access_key", "")
    monkeypatch.setattr(settings, "app_env", "test")
    monkeypatch.setattr(settings, "rerank_enabled", False)
    monkeypatch.setattr(settings, "embedding_provider", "local")
    monkeypatch.setattr(embeddings, "_embed_local", _fake_embed)
    monkeypatch.setattr(vector_store, "_client", None)
    monkeypatch.setattr(persona_service, "_cache", {})
    monkeypatch.setattr(chat_service, "_sessions", OrderedDict())
    yield tmp_path


@dataclass
class FakeLLM:
    reply: str = "נכון....."
    calls: list[list[dict]] = field(default_factory=list)

    def __call__(self, messages: list[dict]) -> str:
        self.calls.append(messages)
        return self.reply

    @property
    def last_system(self) -> str:
        return "\n".join(m["content"] for m in self.calls[-1] if m["role"] == "system")


@pytest.fixture
def fake_llm(monkeypatch: pytest.MonkeyPatch) -> FakeLLM:
    llm = FakeLLM()
    monkeypatch.setattr("app.services.chat.generate_response", llm)
    return llm


@pytest.fixture
def persona() -> dict:
    return {
        "display_name": "דוד כהן",
        "overall_impression": "אב משפחה חם וישיר",
        "family_context": {
            "wife": "רחל",
            "children": ["יונתן — הבכור", "נועה — בת"],
            "grandchildren": ["גל", "תמר"],
        },
        "professional_background": {"current_role": "סמנכ\"ל שיווק", "education": "BSc אלקטדודקה"},
        "health_journey": {"note": "internal", "timeline": "טיפולים בשיבא"},
        "topics_to_avoid": ["אל תמציא שמות"],
        "bot_voice": {
            "rules": ["תשובה קצרה."],
            "signature_lines": ["הנשר נחת...."],
            "example_exchanges": [["חזרנו מהטיול", "הנשר נחת.... איך היה?"]],
            "slang": ["\"סבבה\" = בסדר"],
        },
        "retrieval_aliases": {"אמא": "רחל"},
    }


@pytest.fixture
def write_persona(persona: dict):
    def _write(data: dict | None = None) -> Path:
        settings.personas_dir.mkdir(parents=True, exist_ok=True)
        path = settings.personas_dir / f"{PERSON_ID}.json"
        path.write_text(json.dumps(data or persona, ensure_ascii=False), encoding="utf-8")
        return path

    return _write
