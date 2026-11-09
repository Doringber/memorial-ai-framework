"""Rebuild a person's vector index from data/uploads + the memory log.

Drops the archive collection first, so running it twice gives the same result.
Upload files with identical content are indexed once; forgotten memories are
skipped. Also rebuilds the style bank from every WhatsApp export in
data/whatsapp/ (falling back to the export in data/uploads/).

Usage (from apps/api):
    .venv/bin/python -m scripts.reingest <person_id>
    .venv/bin/python -m scripts.reingest <person_id> --whatsapp a.txt b.txt --sender "דוד כהן"
    .venv/bin/python -m scripts.reingest <person_id> --style-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from loguru import logger

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import settings  # noqa: E402
from app.services.embeddings import embed_passages  # noqa: E402
from app.services.ingestion import ingest_document  # noqa: E402
from app.services.memory import UNINDEXED_SOURCES, iter_memories, memory_chunk  # noqa: E402
from app.services.style_bank import build_style_bank, extract_conversation_windows  # noqa: E402
from app.services.vector_store import delete_collection, store_chunks  # noqa: E402

DEFAULT_WHATSAPP = "roni_whatsapp_family_chat.txt"
DEFAULT_SENDER = "דוד כהן"


def unique_uploads(uploads_dir: Path) -> list[tuple[str, Path]]:
    """Return (content_hash, path) for each distinct *.txt upload, first file wins."""
    seen: dict[str, Path] = {}
    for path in sorted(uploads_dir.glob("*.txt")):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        seen.setdefault(digest, path)
    return list(seen.items())


def whatsapp_exports() -> list[Path]:
    """Default style-bank sources: data/whatsapp/*.txt, else the legacy upload."""
    exports = sorted(settings.whatsapp_dir.glob("*.txt"))
    legacy = settings.uploads_dir / DEFAULT_WHATSAPP
    if not exports and legacy.exists():
        exports = [legacy]
    return exports


def reingest_uploads(person_id: str) -> dict:
    files = unique_uploads(settings.uploads_dir)
    summary = {"distinct_files": len(files), "ok": 0, "empty": 0, "failed": 0}
    for i, (digest, path) in enumerate(files, 1):
        try:
            result = ingest_document(
                person_id=person_id,
                file_path=str(path),
                title=path.stem,
                source_type="txt",
                document_id=f"doc_{digest[:16]}",
            )
            summary["ok" if result.get("status") == "ok" else "empty"] += 1
            logger.info("[{}/{}] {} -> {}", i, len(files), path.name, result.get("status"))
        except Exception as e:  # keep going; report at the end
            summary["failed"] += 1
            logger.error("[{}/{}] {} failed: {}", i, len(files), path.name, e)
    return summary


def reingest_memories(person_id: str) -> dict:
    chunks = [
        memory_chunk(person_id, entry)
        for entry in iter_memories(person_id)
        if entry.get("source") not in UNINDEXED_SOURCES
    ]
    if chunks:
        ids, texts, metadatas = (list(column) for column in zip(*chunks))
        store_chunks(person_id, ids, texts, embed_passages(texts), metadatas)
    return {"memories": len(chunks)}


def reingest_whatsapp_windows(person_id: str, exports: list[Path], sender: str) -> dict:
    """Index multi-turn WhatsApp exchanges around `sender`'s messages into the archive collection."""
    by_id: dict[str, dict] = {}
    for path in exports:
        text = path.read_text(encoding="utf-8", errors="replace")
        for win in extract_conversation_windows(text, sender):
            by_id.setdefault(win["id"], win)
    windows = list(by_id.values())
    for start in range(0, len(windows), 128):
        batch = windows[start : start + 128]
        store_chunks(
            person_id=person_id,
            chunk_ids=[w["id"] for w in batch],
            texts=[w["text"] for w in batch],
            embeddings=embed_passages([w["text"] for w in batch]),
            metadatas=[{**w["metadata"], "person_id": person_id} for w in batch],
            kind="archive",
        )
    return {"whatsapp_windows": len(windows)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("person_id")
    parser.add_argument("--whatsapp", type=Path, nargs="+", help="WhatsApp exports (default: data/whatsapp/*.txt)")
    parser.add_argument("--sender", default=DEFAULT_SENDER, help="Sender name as it appears in the export")
    parser.add_argument("--style-only", action="store_true", help="Only rebuild the style bank")
    args = parser.parse_args()
    exports: list[Path] = args.whatsapp or whatsapp_exports()
    missing = [p for p in exports if not p.exists()]
    if missing:
        parser.error(f"WhatsApp export not found: {', '.join(map(str, missing))}")

    if args.style_only:
        count = build_style_bank(args.person_id, exports, args.sender)
        print(json.dumps({"style_messages": count}))
        return 0

    logger.info("Rebuilding index for person {}", args.person_id)
    delete_collection(args.person_id, "archive")

    summary = {
        "uploads": reingest_uploads(args.person_id),
        "memories": reingest_memories(args.person_id),
        "whatsapp_archive": reingest_whatsapp_windows(args.person_id, exports, args.sender) if exports else {},
        "style_messages": build_style_bank(args.person_id, exports, args.sender) if exports else 0,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
