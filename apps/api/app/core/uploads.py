"""Stream an uploaded file to disk with a size limit and a suffix allowlist."""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import HTTPException, UploadFile, status

_CHUNK = 1024 * 1024


def save_upload(upload: UploadFile, dest_dir: Path, max_mb: int, allowed_suffixes: set[str], default_suffix: str) -> Path:
    suffix = (Path(upload.filename or "").suffix or default_suffix).lower()
    if suffix not in allowed_suffixes:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, f"Unsupported file type '{suffix}'")

    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / f"{uuid.uuid4()}{suffix}"
    limit = max_mb * _CHUNK
    written = 0
    try:
        with path.open("wb") as out:
            while chunk := upload.file.read(_CHUNK):
                written += len(chunk)
                if written > limit:
                    raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, f"File exceeds {max_mb} MB")
                out.write(chunk)
    except HTTPException:
        path.unlink(missing_ok=True)
        raise
    return path
