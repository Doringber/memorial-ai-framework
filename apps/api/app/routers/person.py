"""Person management — create, list, and delete memorial persons, plus their photo."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from loguru import logger
from pydantic import BaseModel

from app.core.config import settings
from app.core.security import require_admin, require_family
from app.core.uploads import save_upload
from app.services.persona import draft_path, load_persona, persona_path
from app.services.vector_store import count_chunks, delete_person_data

router = APIRouter(prefix="/api/person", tags=["person"])

_AVATAR_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}


def _valid_person_id(person_id: str) -> str:
    """Person ids are UUIDs; anything else could escape the avatars directory."""
    try:
        return str(uuid.UUID(person_id))
    except ValueError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Unknown person") from None


def _is_image(header: bytes) -> bool:
    return (
        header.startswith(b"\xff\xd8\xff")
        or header.startswith(b"\x89PNG\r\n\x1a\n")
        or (header[:4] == b"RIFF" and header[8:12] == b"WEBP")
    )


def _find_avatar(person_id: str) -> Optional[Path]:
    for suffix in _AVATAR_TYPES:
        path = settings.avatars_dir / f"{person_id}{suffix}"
        if path.is_file():
            return path
    return None


def _load_persons() -> list[dict]:
    if settings.persons_file.exists():
        return json.loads(settings.persons_file.read_text(encoding="utf-8"))
    return []


def _save_persons(persons: list[dict]) -> None:
    settings.persons_file.parent.mkdir(parents=True, exist_ok=True)
    settings.persons_file.write_text(json.dumps(persons, indent=2, ensure_ascii=False), encoding="utf-8")


class PersonCreate(BaseModel):
    display_name: str
    description: str = ""


class PersonOut(BaseModel):
    id: str
    display_name: str
    description: str
    chunk_count: int = 0
    has_persona: bool = False
    has_avatar: bool = False


@router.post("/create", response_model=PersonOut, dependencies=[Depends(require_admin)])
def create_person(req: PersonCreate) -> PersonOut:
    persons = _load_persons()
    person = {"id": str(uuid.uuid4()), "display_name": req.display_name, "description": req.description}
    persons.append(person)
    _save_persons(persons)
    logger.info("Created person {}", person["id"])
    return PersonOut(**person)


@router.get("/list", response_model=list[PersonOut], dependencies=[Depends(require_family)])
def list_persons() -> list[PersonOut]:
    return [
        PersonOut(
            id=p["id"],
            display_name=p["display_name"],
            description=p.get("description", ""),
            chunk_count=count_chunks(p["id"]),
            has_persona=load_persona(p["id"]) is not None,
            has_avatar=_find_avatar(p["id"]) is not None,
        )
        for p in _load_persons()
    ]


@router.get("/{person_id}/avatar", dependencies=[Depends(require_family)])
def get_avatar(person_id: str) -> FileResponse:
    avatar = _find_avatar(_valid_person_id(person_id))
    if avatar is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No photo")
    return FileResponse(
        avatar, media_type=_AVATAR_TYPES[avatar.suffix], headers={"Cache-Control": "private, max-age=3600"}
    )


@router.post("/{person_id}/avatar", dependencies=[Depends(require_admin)])
def upload_avatar(person_id: str, file: UploadFile = File(...)) -> dict:
    """Replace the person's photo. Content is checked, not just the file name."""
    person_id = _valid_person_id(person_id)
    uploaded = save_upload(file, settings.avatars_dir, settings.max_avatar_upload_mb, set(_AVATAR_TYPES), ".jpg")
    with uploaded.open("rb") as f:
        header = f.read(12)
    if not _is_image(header):
        uploaded.unlink(missing_ok=True)
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Not a JPEG, PNG or WebP image")

    while (old := _find_avatar(person_id)) is not None:
        old.unlink()
    uploaded.rename(settings.avatars_dir / f"{person_id}{uploaded.suffix}")
    logger.info("Updated photo for person {}", person_id)
    return {"status": "ok"}


@router.delete("/{person_id}/data", dependencies=[Depends(require_admin)])
def delete_all_data(person_id: str) -> dict:
    """Delete vectors and persona files. Memory logs and uploads are kept for rebuilds."""
    delete_person_data(person_id)
    for path in (persona_path(person_id), draft_path(person_id)):
        path.unlink(missing_ok=True)
    logger.info("Deleted vector + persona data for person {}", person_id)
    return {"status": "deleted", "person_id": person_id}
