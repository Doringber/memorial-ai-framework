from __future__ import annotations

import os
from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Family material and secrets live outside the repo (which sits in a synced
# folder): MEMORIAL_DATA_HOME/.env and MEMORIAL_DATA_HOME/data. In Docker the values
# arrive as env vars instead.
MEMORIAL_DATA_HOME = Path(os.environ.get("MEMORIAL_DATA_HOME", Path.home() / "MemorialData")).expanduser()

# Hosted providers that receive family text or audio.
EXTERNAL_LLM_PROVIDERS = frozenset({"anthropic", "openai", "cursor"})
EXTERNAL_TTS_PROVIDERS = frozenset({"elevenlabs", "gtts"})


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=MEMORIAL_DATA_HOME / ".env", extra="ignore")

    app_env: str = "development"
    log_level: str = "info"

    # Off by default: nothing leaves this machine. Must be set explicitly to
    # use Anthropic/OpenAI/ElevenLabs/gTTS or Ollama ":cloud" models.
    allow_external_services: bool = False
    # Full chat transcripts on disk. Never used for retrieval.
    save_conversation_log: bool = False

    # Comma-separated origins allowed to call the API from a browser.
    cors_origins: str = "http://localhost:3000,http://localhost:3001"

    # Access control. Empty keys disable auth — only allowed outside production.
    family_access_key: str = ""
    admin_access_key: str = ""

    # LLM
    llm_provider: str = "ollama"  # "ollama" (local), "anthropic", or "openai"
    llm_model: str = "qwen3.5:9b"
    ollama_base_url: str = "http://127.0.0.1:11434"
    # The system prompt (persona + style + archive) is ~10k tokens; Ollama's
    # default window would silently cut it.
    ollama_num_ctx: int = 16384
    ollama_timeout_s: int = 180
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    # Generation tuning — short, human-sounding WhatsApp replies.
    # A 9B local model invents Hebrew words at 0.6 and above (see scripts/eval_chat.py).
    llm_temperature: float = 0.4
    llm_max_tokens: int = 220
    llm_frequency_penalty: float = 0.4
    llm_presence_penalty: float = 0.3

    # Retrieval. E5 models need "query:" / "passage:" prefixes (see embeddings.py).
    embedding_provider: str = "local"  # "local" or "openai"
    embedding_model: str = "intfloat/multilingual-e5-base"
    chunk_size: int = 500
    chunk_overlap: int = 50
    retrieval_top_k: int = 3
    rerank_enabled: bool = True
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_pool_size: int = 12
    # Reranked passages scoring below this are left out of the prompt, so a
    # greeting doesn't drag in unrelated archive facts.
    rerank_min_score: float = 0.2
    style_examples_k: int = 8

    # Voice
    stt_provider: str = "whisper"  # "whisper" (local openai-whisper) or "ivrit" (local ivrit-ai/whisper-large-v3-turbo)
    whisper_model_size: str = "small"
    ivrit_whisper_model: str = "ivrit-ai/whisper-large-v3-turbo"
    tts_provider: str = "none"  # "none" until a local engine is chosen; "elevenlabs"/"gtts" are external
    # ElevenLabs gets its own opt-in: it receives reply text and costs credits,
    # and enabling it must not also unlock hosted LLMs.
    allow_elevenlabs_tts: bool = False
    elevenlabs_api_key: str = ""
    elevenlabs_voice_id: str = ""
    # Validated against ElevenLabs studio exports (ivc_s70_sim99_v4):
    # eleven_v4 with stability=0.70, similarity_boost=0.99, speaker_boost=True.
    elevenlabs_model_id: str = "eleven_v4"
    elevenlabs_stability: float = 0.70
    elevenlabs_similarity_boost: float = 0.99
    elevenlabs_speaker_boost: bool = True
    # Characters sent per calendar month; requests beyond it are refused locally.
    elevenlabs_monthly_char_limit: int = 20000
    # Path to the verified authentic reference clip for free local zero-shot cloning.
    local_voice_reference_wav: str = ""
    max_voice_upload_mb: int = 15
    max_document_upload_mb: int = 50
    max_avatar_upload_mb: int = 10
    audio_retention_hours: int = 24

    # Telegram Bot (optional — works via background long-polling or /api/telegram/webhook)
    telegram_bot_token: str = ""
    telegram_polling_enabled: bool = True
    telegram_webhook_secret: str = ""
    telegram_allowed_chat_ids: str = ""  # comma-separated Telegram chat_ids pre-approved for family access
    telegram_default_person_id: str = ""
    telegram_voice_replies: str = "voice_only"  # "voice_only" (on voice notes or /voice), "always", or "off"

    # Storage
    data_dir: Path = MEMORIAL_DATA_HOME / "data"
    chroma_persist_dir: Path = MEMORIAL_DATA_HOME / "data" / "chroma"

    @model_validator(mode="after")
    def _require_keys_in_production(self) -> Settings:
        if self.is_production and not (self.family_access_key and self.admin_access_key):
            raise ValueError(
                "FAMILY_ACCESS_KEY and ADMIN_ACCESS_KEY must be set when APP_ENV=production"
            )
        return self

    @model_validator(mode="after")
    def _block_external_services(self) -> Settings:
        if self.allow_external_services:
            return self
        external = [
            name
            for name, used in (
                (f"LLM_PROVIDER={self.llm_provider}", self.llm_provider in EXTERNAL_LLM_PROVIDERS),
                (f"LLM_MODEL={self.llm_model}", is_ollama_cloud_model(self.llm_model)),
                (f"TTS_PROVIDER={self.tts_provider}", not self.tts_allowed),
                (f"EMBEDDING_PROVIDER={self.embedding_provider}", self.embedding_provider == "openai"),
            )
            if used
        ]
        if external:
            raise ValueError(
                f"{', '.join(external)} sends family data off this machine. "
                "Switch to a local provider or set ALLOW_EXTERNAL_SERVICES=true explicitly "
                "(ALLOW_ELEVENLABS_TTS=true covers ElevenLabs voice only)."
            )
        return self

    @property
    def tts_allowed(self) -> bool:
        """Whether the configured TTS provider may be used under the consent flags."""
        if self.tts_provider not in EXTERNAL_TTS_PROVIDERS or self.allow_external_services:
            return True
        return self.tts_provider == "elevenlabs" and self.allow_elevenlabs_tts

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def personas_dir(self) -> Path:
        return self.data_dir / "personas"

    @property
    def memories_dir(self) -> Path:
        return self.data_dir / "memories"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def whatsapp_dir(self) -> Path:
        """WhatsApp exports for the style bank. Kept out of uploads so whole
        family chats don't flood the memory archive."""
        return self.data_dir / "whatsapp"

    @property
    def avatars_dir(self) -> Path:
        return self.data_dir / "avatars"

    @property
    def audio_output_dir(self) -> Path:
        return self.data_dir / "audio_output"

    @property
    def usage_dir(self) -> Path:
        return self.data_dir / "usage"

    @property
    def voice_uploads_dir(self) -> Path:
        return self.data_dir / "voice_uploads"

    @property
    def persons_file(self) -> Path:
        return self.data_dir / "persons.json"


def is_ollama_cloud_model(model: str) -> bool:
    """Ollama ":cloud" / "-cloud" tags run on Ollama's servers, not locally."""
    return model.endswith(":cloud") or model.endswith("-cloud")


settings = Settings()

# Models are already in the local cache; never phone home at runtime.
for _var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
    os.environ.setdefault(_var, "1")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
