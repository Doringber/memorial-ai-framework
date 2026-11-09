from __future__ import annotations

import json
from collections.abc import Iterator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.core.security import require_family
from app.models.schemas import (
    ChatStreamRequest,
    ChatTextRequest,
    ChatTextResponse,
    LatencyReportRequest,
    SpokenCommitRequest,
    SpokenCommitResponse,
)
from app.services.chat import chat_text, commit_spoken_reply, log_latency_metrics, stream_chat_text

router = APIRouter(prefix="/api/chat", tags=["chat"], dependencies=[Depends(require_family)])


def _ndjson(events: Iterator[dict]) -> Iterator[bytes]:
    for event in events:
        yield (json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8")


# Sync handler on purpose: retrieval, embeddings and the LLM call block, so
# FastAPI runs this in its threadpool instead of on the event loop.
@router.post("/text", response_model=ChatTextResponse)
def handle_chat_text(req: ChatTextRequest) -> ChatTextResponse:
    result = chat_text(
        person_id=str(req.person_id),
        message=req.message,
        session_id=str(req.session_id) if req.session_id else None,
    )
    return ChatTextResponse(**result)


@router.post("/stream")
def handle_chat_stream(req: ChatStreamRequest) -> StreamingResponse:
    """Stream a chat reply sentence by sentence as NDJSON events."""
    events = stream_chat_text(
        person_id=str(req.person_id),
        message=req.message,
        session_id=str(req.session_id) if req.session_id else None,
        defer_history=req.defer_history,
    )
    return StreamingResponse(_ndjson(events), media_type="application/x-ndjson")


@router.post("/spoken", response_model=SpokenCommitResponse)
def handle_commit_spoken(req: SpokenCommitRequest) -> SpokenCommitResponse:
    """Record in session history only the prefix of a reply that was actually played."""
    result = commit_spoken_reply(
        session_id=str(req.session_id),
        user_message=req.user_message,
        spoken_text=req.spoken_text,
    )
    return SpokenCommitResponse(**result)


@router.post("/latency")
def handle_latency_report(req: LatencyReportRequest) -> dict[str, str]:
    """Record privacy-safe turn latency measurements (numbers only, no text)."""
    log_latency_metrics(req.model_dump())
    return {"status": "ok"}
