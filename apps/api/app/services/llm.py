"""LLM provider abstraction.

Supports:
- "ollama"    — local Ollama, native /api/chat (default; nothing leaves the machine)
- "anthropic" — Anthropic API (Claude). External: needs ALLOW_EXTERNAL_SERVICES.
- "openai"    — OpenAI API. External: needs ALLOW_EXTERNAL_SERVICES.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from urllib.parse import urlparse

import httpx
from loguru import logger

from app.core.config import EXTERNAL_LLM_PROVIDERS, is_ollama_cloud_model, settings

_RATE_LIMIT_WAITS = (5, 15, 30)
# host.docker.internal is the host machine as seen from a local container.
_LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "host.docker.internal"})
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
# Match sentence-ending punctuation (plus any trailing emoji/spaces) only once
# the next sentence has started, so "...." + emoji stays together in one chunk.
_SENTENCE_BOUNDARY = re.compile(
    "(?:[.!?…]+[\\s\U0001F300-\U0001FAFF\u2600-\u27BF]*\\s+|\n+)"
    "(?=[^\\s.!?…\U0001F300-\U0001FAFF\u2600-\u27BF])"
)
_REPLY_EMOJI = re.compile("[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0000FE0F\U0000200D]+")
_MIN_SENTENCE_CHARS = 10
_MAX_CLAUSE_CHARS = 90


def clean_reply_emojis(text: str, max_emojis: int = 1) -> str:
    """Keep at most `max_emojis` emoji characters in `text` so replies don't end with emoji chains."""
    seen = 0
    out: list[str] = []
    idx = 0
    for match in _REPLY_EMOJI.finditer(text):
        out.append(text[idx : match.start()])
        cluster = match.group(0)
        # Count visible emoji codepoints (ignoring variation selectors / ZWJ).
        visible = [c for c in cluster if ord(c) not in (0xFE0F, 0x200D)]
        for ch in visible:
            if seen < max_emojis:
                out.append(ch)
                seen += 1
        idx = match.end()
    out.append(text[idx:])
    return re.sub(r" {2,}", " ", "".join(out)).strip()


class ExternalServiceBlockedError(RuntimeError):
    """Raised when a call would send family data to a hosted service without consent."""


def _ensure_allowed() -> None:
    if settings.allow_external_services:
        return
    if settings.llm_provider in EXTERNAL_LLM_PROVIDERS:
        raise ExternalServiceBlockedError(f"LLM provider '{settings.llm_provider}' is external")
    if settings.llm_provider == "ollama":
        if is_ollama_cloud_model(settings.llm_model):
            raise ExternalServiceBlockedError(f"Ollama model '{settings.llm_model}' runs in the cloud")
        host = urlparse(settings.ollama_base_url).hostname
        if host not in _LOCAL_HOSTS:
            raise ExternalServiceBlockedError(f"Ollama host '{host}' is not this machine")


def generate_response(messages: list[dict]) -> str:
    """Send OpenAI-format messages to the configured LLM and return the reply text."""
    _ensure_allowed()
    if settings.llm_provider == "ollama":
        return _generate_ollama(messages)
    if settings.llm_provider == "cursor":
        return _generate_cursor(messages)
    if settings.llm_provider == "anthropic":
        return _generate_anthropic(messages)
    return _generate_openai(messages)


def stream_response(messages: list[dict]) -> Iterator[str]:
    """Yield text deltas from the configured LLM."""
    _ensure_allowed()
    if settings.llm_provider == "ollama":
        yield from _stream_ollama(messages)
        return
    if settings.llm_provider == "cursor":
        yield from _stream_cursor(messages)
        return
    reply = generate_response(messages)
    if reply:
        yield reply


def iter_sentences(chunks: Iterable[str]) -> Iterator[str]:
    """Buffer streaming token deltas and yield spoken sentence chunks as early as possible."""
    buffer = ""
    for chunk in chunks:
        if not chunk:
            continue
        buffer += chunk
        while True:
            split_at = _find_sentence_split(buffer)
            if split_at is None:
                break
            sentence = buffer[:split_at].strip()
            buffer = buffer[split_at:].lstrip()
            if sentence:
                yield sentence
    tail = buffer.strip()
    if tail:
        yield tail


def _find_sentence_split(buffer: str) -> int | None:
    """Return index in `buffer` where a spoken sentence ends, or None if more tokens are needed."""
    for match in _SENTENCE_BOUNDARY.finditer(buffer):
        end = match.end()
        if len(buffer[:end].strip()) >= _MIN_SENTENCE_CHARS:
            return end
    if len(buffer) >= _MAX_CLAUSE_CHARS:
        for sep in (", ", " — ", " - ", " "):
            idx = buffer.rfind(sep, _MIN_SENTENCE_CHARS, _MAX_CLAUSE_CHARS)
            if idx != -1:
                return idx + len(sep)
    return None


def _ollama_payload(messages: list[dict], *, stream: bool) -> dict:
    return {
        "model": settings.llm_model,
        "messages": messages,
        "stream": stream,
        "think": False,
        "options": {
            "num_ctx": settings.ollama_num_ctx,
            "num_predict": settings.llm_max_tokens,
            "temperature": settings.llm_temperature,
            "repeat_penalty": 1.0 + settings.llm_frequency_penalty / 2,
        },
    }


def _stream_ollama(messages: list[dict]) -> Iterator[str]:
    logger.debug("Streaming Ollama {} with {} messages", settings.llm_model, len(messages))
    in_think = False
    carry = ""
    with httpx.stream(
        "POST",
        f"{settings.ollama_base_url}/api/chat",
        json=_ollama_payload(messages, stream=True),
        timeout=settings.ollama_timeout_s,
    ) as response:
        response.raise_for_status()
        for raw_line in response.iter_lines():
            if not raw_line:
                continue
            data = json.loads(raw_line)
            piece = data.get("message", {}).get("content", "")
            if piece:
                carry += piece
                carry, in_think, clean = _strip_streaming_think(carry, in_think)
                if clean:
                    yield clean
            if data.get("done"):
                prompt_tokens = data.get("prompt_eval_count", 0)
                if prompt_tokens >= settings.ollama_num_ctx:
                    logger.warning(
                        "Prompt filled the whole context ({} tokens); it may be truncated",
                        prompt_tokens,
                    )
                logger.info(
                    "Ollama stream {}: {} prompt tokens, {} reply tokens, {:.1f}s",
                    settings.llm_model,
                    prompt_tokens,
                    data.get("eval_count", 0),
                    data.get("total_duration", 0) / 1e9,
                )
    if carry and not in_think:
        cleaned = _THINK_BLOCK.sub("", carry)
        if cleaned:
            yield cleaned


def _strip_streaming_think(text: str, in_think: bool) -> tuple[str, bool, str]:
    """Strip `<think>...</think>` blocks that may span multiple streamed deltas."""
    out: list[str] = []
    while text:
        if in_think:
            end = text.find("</think>")
            if end == -1:
                # Keep tail in case `</think>` is split across chunks.
                return text[-8:], True, "".join(out)
            text = text[end + len("</think>") :]
            in_think = False
        else:
            start = text.find("<think>")
            if start == -1:
                # Hold back a trailing `<` prefix in case `<think>` is starting.
                for prefix_len in range(min(len("<think>") - 1, len(text)), 0, -1):
                    if "<think>".startswith(text[-prefix_len:]):
                        out.append(text[:-prefix_len])
                        return text[-prefix_len:], False, "".join(out)
                out.append(text)
                return "", False, "".join(out)
            out.append(text[:start])
            text = text[start + len("<think>") :]
            in_think = True
    return "", in_think, "".join(out)


def _generate_ollama(messages: list[dict]) -> str:
    logger.debug("Calling Ollama {} with {} messages", settings.llm_model, len(messages))
    response = httpx.post(
        f"{settings.ollama_base_url}/api/chat",
        json=_ollama_payload(messages, stream=False),
        timeout=settings.ollama_timeout_s,
    )
    response.raise_for_status()
    data = response.json()
    prompt_tokens = data.get("prompt_eval_count", 0)
    if prompt_tokens >= settings.ollama_num_ctx:
        logger.warning("Prompt filled the whole context ({} tokens); it may be truncated", prompt_tokens)
    logger.info(
        "Ollama {}: {} prompt tokens, {} reply tokens, {:.1f}s",
        settings.llm_model,
        prompt_tokens,
        data.get("eval_count", 0),
        data.get("total_duration", 0) / 1e9,
    )
    return _THINK_BLOCK.sub("", data.get("message", {}).get("content", "")).strip()


def _prepare_cursor_env() -> tuple[dict[str, str], str]:
    """Create a minimal Cursor home without MCP servers or hooks so CLI startup is sub-second."""
    import os
    from pathlib import Path

    real_home = Path.home()
    cursor_home = Path("/tmp/askronny-cursor-home")
    cursor_dir = cursor_home / ".cursor"
    empty_ws = Path("/tmp/askronny-empty-ws")
    cursor_dir.mkdir(parents=True, exist_ok=True)
    empty_ws.mkdir(parents=True, exist_ok=True)

    lib_link = cursor_home / "Library"
    if not lib_link.exists():
        lib_link.symlink_to(real_home / "Library")

    for fname in ("cli-config.json", "agent-cli-state.json", "ide_state.json"):
        src = real_home / ".cursor" / fname
        dst = cursor_dir / fname
        if src.exists() and not dst.exists():
            dst.symlink_to(src)

    (cursor_dir / "mcp.json").write_text('{"mcpServers":{}}', encoding="utf-8")
    (cursor_dir / "hooks.json").write_text('{"hooks":{}}', encoding="utf-8")

    env = dict(os.environ)
    env["HOME"] = str(cursor_home)
    return env, str(empty_ws)


def _format_cursor_prompt(messages: list[dict]) -> str:
    header = (
        "חשוב: אל תפעל כסוכן קוד או יועץ רשמי, ואל תזכיר את מצב Ask, כלים, חיפושים או קבצים. "
        "ענה אך ורק בדמות המתוארת ב-[SYSTEM] בהודעת וואטסאפ קצרה וטבעית (שורה או שתיים, לרוב 10–25 מילים), "
        "באותו טון, אוצר מילים וקצב כתיבה שבדוגמאות הוואטסאפ שלה. "
        "אף פעם אל תכתוב הרצאה או רשימת שלבים — תן ישר את השורה התחתונה בעברית בלבד וללא מרכאות עוטפות."
    )
    parts = [header]
    for msg in messages:
        role = msg["role"].upper()
        parts.append(f"[{role}]\n{msg['content']}")
    return "\n\n".join(parts)


def _stream_cursor(messages: list[dict]) -> Iterator[str]:
    """Stream text deltas from `cursor-agent` CLI and terminate immediately on the `result` event."""
    import shutil
    import subprocess

    binary = shutil.which("cursor-agent") or str(Path.home() / ".local" / "bin" / "cursor-agent")
    env, empty_ws = _prepare_cursor_env()
    cmd = [
        binary,
        "-p",
        "--trust",
        "--workspace",
        empty_ws,
        "--mode",
        "ask",
        "--output-format",
        "stream-json",
        "--stream-partial-output",
    ]
    if settings.llm_model and not settings.llm_model.startswith("qwen"):
        cmd.extend(["--model", settings.llm_model])
    cmd.append(_format_cursor_prompt(messages))

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    yielded_any = False
    try:
        assert proc.stdout is not None
        for raw_line in proc.stdout:
            line = raw_line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            etype = event.get("type")
            if etype == "assistant" and "timestamp_ms" in event:
                for block in event.get("message", {}).get("content", []):
                    if block.get("type") == "text" and block.get("text"):
                        yielded_any = True
                        yield block["text"]
            elif etype == "result":
                if not yielded_any and event.get("result"):
                    yield str(event["result"])
                break
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()


def _generate_cursor(messages: list[dict]) -> str:
    """Run prompt through local `cursor-agent` CLI in read-only ask mode (external cloud model)."""
    return _THINK_BLOCK.sub("", "".join(_stream_cursor(messages))).strip()


def _generate_openai(messages: list[dict]) -> str:
    from openai import OpenAI

    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY not set. Cannot generate responses.")
    client = OpenAI(api_key=settings.openai_api_key)

    logger.debug("Calling OpenAI {} with {} messages", settings.llm_model, len(messages))
    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        frequency_penalty=settings.llm_frequency_penalty,
        presence_penalty=settings.llm_presence_penalty,
    )
    return response.choices[0].message.content or ""


def _generate_anthropic(messages: list[dict]) -> str:
    from anthropic import Anthropic, RateLimitError

    if not settings.anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set. Cannot generate responses.")

    client = Anthropic(api_key=settings.anthropic_api_key, timeout=120.0)

    system_blocks: list[dict] = []
    chat_messages: list[dict] = []
    for msg in messages:
        if msg["role"] == "system":
            system_blocks.append({"type": "text", "text": msg["content"]})
        else:
            chat_messages.append({"role": msg["role"], "content": msg["content"]})
    if system_blocks:
        system_blocks[0]["cache_control"] = {"type": "ephemeral"}

    logger.debug("Calling Anthropic {} with {} messages", settings.llm_model, len(chat_messages))

    for attempt in range(len(_RATE_LIMIT_WAITS) + 1):
        try:
            response = client.messages.create(
                model=settings.llm_model,
                max_tokens=settings.llm_max_tokens,
                system=system_blocks,
                messages=chat_messages,
                temperature=settings.llm_temperature,
            )
            return "".join(block.text for block in response.content if block.type == "text")
        except RateLimitError:
            if attempt == len(_RATE_LIMIT_WAITS):
                raise
            wait = _RATE_LIMIT_WAITS[attempt]
            logger.warning("Anthropic rate limit; retry {} in {}s", attempt + 1, wait)
            time.sleep(wait)
    raise AssertionError("unreachable")
