from app.core.config import settings
from app.services.memory import forget_facts, save_memory
from app.services.vector_store import count_chunks, delete_collection
from scripts.reingest import (
    DEFAULT_WHATSAPP,
    reingest_memories,
    reingest_uploads,
    unique_uploads,
    whatsapp_exports,
)
from tests.conftest import PERSON_ID


def _write_uploads(files: dict[str, str]) -> None:
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (settings.uploads_dir / name).write_text(text, encoding="utf-8")


def test_unique_uploads_dedupes_by_content() -> None:
    _write_uploads({"a.txt": "זהה", "b.txt": "זהה", "c.txt": "שונה", "d.mp4": "זהה"})
    names = sorted(p.name for _, p in unique_uploads(settings.uploads_dir))
    assert names == ["a.txt", "c.txt"]


def test_whatsapp_exports_prefers_whatsapp_dir_over_legacy_upload() -> None:
    _write_uploads({DEFAULT_WHATSAPP: "legacy"})
    assert whatsapp_exports() == [settings.uploads_dir / DEFAULT_WHATSAPP]

    settings.whatsapp_dir.mkdir(parents=True)
    for name in ("b.txt", "a.txt"):
        (settings.whatsapp_dir / name).write_text("x", encoding="utf-8")
    assert [p.name for p in whatsapp_exports()] == ["a.txt", "b.txt"]


def test_reingest_is_idempotent_and_skips_forgotten() -> None:
    _write_uploads({"a.txt": "דוד גדל בבת ים", "b.txt": "דוד גדל בבת ים"})
    save_memory(PERSON_ID, "עובדה שנשארת", source="explicit_save")
    save_memory(PERSON_ID, "עובדה שנשכחת", source="explicit_save")
    forget_facts(PERSON_ID, "שנשכחת")

    for _ in range(2):
        delete_collection(PERSON_ID, "archive")
        assert reingest_uploads(PERSON_ID)["ok"] == 1
        assert reingest_memories(PERSON_ID) == {"memories": 1}
        assert count_chunks(PERSON_ID) == 2
