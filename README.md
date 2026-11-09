# Living Memory (זיכרון חי) — Open-Source AI Memorial Framework

A privacy-first, open-source framework for families to preserve the writing style, stories, memories, and voice of a loved one who has passed away — accessible via a **Web Chat UI** and a **Telegram Bot** (text + voice notes).

---

## Core Principles & Built-In Ethics

1. **Explicit AI Transparency:** Every screen and Telegram welcome message clearly states that this is an AI simulation built from archived messages and recordings — never the person themselves.
2. **Strict Data Separation:** Personal family data (WhatsApp exports, photos, voice samples, Chroma vectors, and `.env` secrets) lives in a private directory outside the repository (`~/MemorialData` or `$MEMORIAL_DATA_HOME` with `0700` permissions) so family data is never committed to Git.
3. **Built-In Crisis & Directive Safety (`app/services/safety.py`):**
   - **Self-harm / grief crisis detection:** Immediately steps out of persona and provides emergency support lines (e.g., ERAN 1201 in Israel).
   - **Financial / legal / impersonation guard:** Refuses requests about wills, inheritance, passwords, or sending messages to third parties in the deceased person's name.
4. **Honesty Guard:** Never fabricates biographical or family memories. If a specific personal memory is not in the archive, the bot says so naturally.

---

## Key Features

- **WhatsApp Style Bank & Conversation-Window RAG (`app/services/style_bank.py`):**
  Upload a WhatsApp `_chat.txt` export in `/admin`. The system automatically detects all participants, extracts the loved one's real reply patterns (`prompt -> reply`), and indexes multi-turn family conversation windows into ChromaDB (`intfloat/multilingual-e5-base` + `BAAI/bge-reranker-v2-m3`).
- **One-Click Persona Builder (`/admin`):**
  Analyzes the loved one's messages to draft and activate a structured `persona.json` (common phrases, humor, tone, family context, and topics of knowledge), or lets you curate one from `examples/persona.example.json`.
- **Voice Conversation & Barge-In:**
  - **Speech-to-Text (STT):** Local Whisper (`small`) or `ivrit-ai/whisper-large-v3-turbo` (optimized for Israeli Hebrew on Apple Silicon MPS).
  - **Text-to-Speech (TTS):** Opt-in **ElevenLabs v4** Instant Voice Clone (`eleven_v4`) with hard monthly character budgets and zero automatic retries, or **free local zero-shot voice cloning** via Chatterbox Multilingual.
  - **Sentence-by-Sentence Streaming & Barge-In:** Plays the first sentence while the rest streams; interrupting stops playback immediately and commits **only what was actually heard** to conversation history.
- **Telegram Bot (`app/services/telegram_bot.py`):**
  Works out of the box via background long-polling (no port-forwarding required) or HTTPS webhook. Supports text replies, inline `🔊 השמע בקול` buttons, `/voice` auto-voice mode, and incoming/outgoing Opus voice notes.

---

## Quick Start (10 Minutes)

### 1. Configure Private Data Directory
```bash
mkdir -p ~/MemorialData && chmod 700 ~/MemorialData
cp .env.example ~/MemorialData/.env && chmod 600 ~/MemorialData/.env
```

### 2. Start Backend & Frontend
```bash
# Backend (FastAPI on 127.0.0.1:8000)
cd apps/api
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
./.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000

# Frontend (Next.js on 127.0.0.1:3000)
cd apps/web
npm install
npm run dev
```

### 3. Onboard Your Loved One in `/admin`
1. Open **http://127.0.0.1:3000/admin**.
2. Create the person's profile and upload their photo.
3. Under **ייבוא שיחת וואטסאפ (`_chat.txt`)**, select a exported WhatsApp chat file, choose the person's name from the detected senders dropdown, and click **בנה סגנון ושיחות**.
4. Click **צור והפעל פרופיל אישיות** (or customize `$MEMORIAL_DATA_HOME/data/personas/<person_id>.json` based on `examples/persona.example.json`).

### 4. Optional: Connect a Telegram Bot
1. Message [`@BotFather`](https://t.me/BotFather) on Telegram -> `/newbot` -> copy the token.
2. Add `TELEGRAM_BOT_TOKEN=<token>` to `~/MemorialData/.env` and restart the API.
3. Share `https://t.me/<your_bot_username>` with your family.

---

## Testing & Evaluation
```bash
# Run unit & integration test suite
cd apps/api && ./.venv/bin/python -m pytest -q

# Replay Golden Set conversations against the chat pipeline
cd apps/api && ./.venv/bin/python -m scripts.eval_chat <person_id> --cases ~/MemorialData/eval/golden_set.json
```

## License
MIT — see [LICENSE](LICENSE).
