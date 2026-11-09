from __future__ import annotations

import hashlib

from fastapi import APIRouter, Depends, File, Form, UploadFile
from pydantic import BaseModel

from app.core.config import settings
from app.core.security import require_admin
from app.core.uploads import save_upload
from app.services.embeddings import embed_passages
from app.services.ingestion import ingest_document
from app.services.style_bank import (
    build_style_bank,
    extract_conversation_windows,
    list_whatsapp_senders,
)
from app.services.vector_store import store_chunks

router = APIRouter(prefix="/api/ingest", tags=["ingestion"], dependencies=[Depends(require_admin)])

_DOCUMENT_SUFFIXES = {".txt", ".md", ".pdf", ".docx", ".doc", ".eml"}
_WHATSAPP_SUFFIXES = {".txt"}


class IngestResponse(BaseModel):
    status: str
    document_id: str = ""
    title: str = ""
    chunks: int = 0
    characters: int = 0


class WhatsAppSenderOut(BaseModel):
    sender: str
    messages: int


class WhatsAppIngestResponse(BaseModel):
    status: str
    sender: str
    style_examples: int
    conversation_windows: int


@router.post("/document", response_model=IngestResponse)
def upload_and_ingest(
    file: UploadFile = File(...),
    person_id: str = Form(...),
    title: str = Form(""),
    source_type: str = Form(""),
) -> IngestResponse:
    """Upload a document, extract text, chunk, embed, and store."""
    save_path = save_upload(
        file, settings.uploads_dir, settings.max_document_upload_mb, _DOCUMENT_SUFFIXES, ".txt"
    )
    result = ingest_document(
        person_id=person_id,
        file_path=str(save_path),
        title=title or file.filename or "untitled",
        source_type=source_type,
    )
    return IngestResponse(**result)


@router.post("/whatsapp-senders", response_model=list[WhatsAppSenderOut])
def detect_whatsapp_senders(file: UploadFile = File(...)) -> list[WhatsAppSenderOut]:
    """Parse an uploaded WhatsApp `_chat.txt` export and return all senders ranked by message count."""
    tmp_path = save_upload(
        file, settings.whatsapp_dir, settings.max_document_upload_mb, _WHATSAPP_SUFFIXES, ".txt"
    )
    try:
        text = tmp_path.read_text(encoding="utf-8", errors="replace")
        return [WhatsAppSenderOut(**item) for item in list_whatsapp_senders(text)]
    finally:
        tmp_path.unlink(missing_ok=True)


@router.post("/whatsapp", response_model=WhatsAppIngestResponse)
def upload_and_ingest_whatsapp(
    file: UploadFile = File(...),
    person_id: str = Form(...),
    sender: str = Form(...),
) -> WhatsAppIngestResponse:
    """Save a WhatsApp export, rebuild the person's style bank, and index multi-turn conversation windows."""
    saved_path = save_upload(
        file, settings.whatsapp_dir, settings.max_document_upload_mb, _WHATSAPP_SUFFIXES, ".txt"
    )
    digest = hashlib.sha256(saved_path.read_bytes()).hexdigest()[:12]
    canonical_path = settings.whatsapp_dir / f"{person_id}_{digest}.txt"
    saved_path.rename(canonical_path)

    exports = sorted(settings.whatsapp_dir.glob(f"{person_id}_*.txt")) or [canonical_path]
    style_count = build_style_bank(person_id, exports, sender.strip())

    text = canonical_path.read_text(encoding="utf-8", errors="replace")
    windows = extract_conversation_windows(text, sender.strip())
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

    return WhatsAppIngestResponse(
        status="ok",
        sender=sender.strip(),
        style_examples=style_count,
        conversation_windows=len(windows),
    )
