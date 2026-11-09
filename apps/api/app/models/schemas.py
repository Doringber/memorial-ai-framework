from uuid import UUID

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str


class ChatTextRequest(BaseModel):
    person_id: UUID
    session_id: UUID | None = None
    message: str = Field(min_length=1, max_length=4000)


class ChatStreamRequest(ChatTextRequest):
    defer_history: bool = False


class SpokenCommitRequest(BaseModel):
    session_id: UUID
    user_message: str = Field(min_length=1, max_length=4000)
    spoken_text: str = Field(default="", max_length=4000)


class SpokenCommitResponse(BaseModel):
    session_id: UUID
    committed_chars: int


class LatencyReportRequest(BaseModel):
    mode: str = Field(default="voice", pattern="^(voice|text)$")
    stt_ms: float | None = None
    rag_ms: float | None = None
    llm_first_sentence_ms: float | None = None
    llm_total_ms: float | None = None
    tts_first_audio_ms: float | None = None
    client_e2e_ms: float | None = None
    interrupted: bool = False
    sentence_count: int | None = None
    spoken_sentences: int | None = None
    reply_chars: int | None = None
    spoken_chars: int | None = None


class Citation(BaseModel):
    document_title: str
    chunk_id: str


class ChatTextResponse(BaseModel):
    reply: str
    citations: list[Citation] = []
    safety_notes: list[str] = []
    session_id: UUID
