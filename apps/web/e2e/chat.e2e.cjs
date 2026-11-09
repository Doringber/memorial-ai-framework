#!/usr/bin/env node
/**
 * End-to-end check of the chat in a real Chromium: text question, "השמע"
 * playback, and push-to-talk (fake microphone fed from a local WAV).
 *
 * E2E_MODE=mock  Fakes the API. No family data, so screenshots are safe to share.
 * E2E_MODE=live  Uses the local API on 127.0.0.1:8000. Prints only timings and
 *                pass/fail, never reply text. Screenshots go to
 *                $MEMORIAL_DATA_HOME/eval/e2e and may contain family material.
 *
 * Live mode with ElevenLabs sends two reply texts to ElevenLabs (a few hundred
 * characters of the monthly budget).
 *
 * Env: E2E_QUESTION_WAV (required; 16-bit PCM WAV with a spoken Hebrew question),
 *      E2E_EXPECTED_TRANSCRIPT (optional substring check), E2E_ACCESS_KEY
 *      (defaults to FAMILY_ACCESS_KEY in $MEMORIAL_DATA_HOME/.env), E2E_BASE_URL,
 *      E2E_API_URL, E2E_OUT_DIR, E2E_HEADED=1.
 */
"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const { chromium } = require("playwright");

const MODE = process.env.E2E_MODE ?? "mock";
const BASE_URL = process.env.E2E_BASE_URL ?? "http://127.0.0.1:3000";
const API_URL = process.env.E2E_API_URL ?? "http://127.0.0.1:8000";
const MEMORIAL_DATA_HOME = process.env.MEMORIAL_DATA_HOME ?? path.join(os.homedir(), "MemorialData");
const OUT_DIR =
  process.env.E2E_OUT_DIR ??
  (MODE === "live" ? path.join(MEMORIAL_DATA_HOME, "eval", "e2e") : path.join(os.tmpdir(), "askronny-e2e", "shots"));
const QUESTION_WAV = process.env.E2E_QUESTION_WAV ?? "";
const EXPECTED_TRANSCRIPT = process.env.E2E_EXPECTED_TRANSCRIPT ?? "";
const TEXT_QUESTION = "מה שלומך היום?";
const RECORD_SECONDS = 5;
const HEBREW = /[\u0590-\u05FF]/;

const TIMEOUTS = MODE === "live" ? { reply: 240_000, speak: 90_000, voice: 360_000 } : { reply: 10_000, speak: 10_000, voice: 10_000 };

function readEnvFileKey(name) {
  const envPath = path.join(MEMORIAL_DATA_HOME, ".env");
  if (!fs.existsSync(envPath)) return "";
  const line = fs
    .readFileSync(envPath, "utf8")
    .split("\n")
    .find((l) => l.startsWith(`${name}=`));
  return line ? line.slice(name.length + 1).trim().replace(/^["']|["']$/g, "") : "";
}

class ChatPage {
  constructor(page) {
    this.page = page;
    this.accessKey = page.getByTestId("access-key");
    this.notice = page.getByTestId("ai-notice");
    this.voiceLabel = page.getByTestId("voice-label");
    this.input = page.getByTestId("message-input");
    this.send = page.getByTestId("send");
    this.record = page.getByTestId("record");
    this.typing = page.getByTestId("typing");
    this.userMessages = page.getByTestId("message-user");
    this.replies = page.getByTestId("message-assistant");
  }

  async open() {
    await this.page.goto(BASE_URL, { waitUntil: "networkidle" });
    await this.accessKey.or(this.notice).waitFor();
  }

  async signIn(key) {
    if (!(await this.accessKey.isVisible())) return;
    assert.ok(key, "Login required: set E2E_ACCESS_KEY or FAMILY_ACCESS_KEY in the data .env");
    await this.accessKey.fill(key);
    await this.page.getByTestId("sign-in").click();
    await this.notice.waitFor();
  }

  reply(index) {
    return this.replies.nth(index);
  }

  async ask(text, timeout) {
    const before = await this.replies.count();
    await this.input.fill(text);
    await this.send.click();
    await this.reply(before).waitFor({ timeout });
    return this.reply(before);
  }

  async speak(reply, timeout) {
    await reply.getByTestId("speak").click();
    await reply.getByTestId("reply-audio").or(reply.getByTestId("voice-unavailable")).waitFor({ timeout });
  }

  async askByVoice(seconds, timeout) {
    const before = await this.replies.count();
    await this.record.click();
    await this.page.locator('[data-testid="record"][aria-pressed="true"]').waitFor();
    await this.record.filter({ hasText: `0:${String(seconds).padStart(2, "0")}` }).waitFor({ timeout: (seconds + 5) * 1000 });
    await this.record.click();
    await this.reply(before).waitFor({ timeout });
    return this.reply(before);
  }

  async shot(name) {
    const file = path.join(OUT_DIR, `${MODE}-${name}.png`);
    await this.page.screenshot({ path: file, fullPage: true });
    return file;
  }
}

/** Waits until an <audio> element has decoded metadata and started playing. */
async function checkAudio(page, reply) {
  const audio = reply.getByTestId("reply-audio");
  if (!(await audio.isVisible())) {
    return { ok: false, reason: (await reply.getByTestId("voice-unavailable").textContent()) ?? "missing" };
  }
  const handle = await audio.elementHandle();
  await page.waitForFunction((el) => el.readyState >= 1 && el.duration > 0, handle, { timeout: 30_000 });
  await page.waitForFunction((el) => el.currentTime > 0 || el.ended, handle, { timeout: 30_000 });
  const stats = await handle.evaluate(async (el) => {
    const res = await fetch(el.src);
    const bytes = (await res.arrayBuffer()).byteLength;
    return { status: res.status, type: res.headers.get("content-type"), bytes, seconds: el.duration, played: el.currentTime > 0 || el.ended };
  });
  const ok = stats.status === 200 && stats.type === "audio/mpeg" && stats.bytes > 1000 && stats.seconds > 0.3 && stats.played;
  return { ok, ...stats, seconds: Number(stats.seconds.toFixed(2)) };
}

async function voiceUsage(page) {
  return page.evaluate(async (api) => {
    const res = await fetch(`${api}/api/chat/voice-info`, {
      headers: { "X-Access-Key": localStorage.getItem("rony.accessKey") ?? "" },
    });
    return res.ok ? res.json() : null;
  }, API_URL);
}

// ---------- mock API (no family data) ----------

const MOCK_KEY = "mock-family-key";
const MOCK_PERSON = { id: "p-mock", display_name: "דוד", description: "", chunk_count: 1, has_persona: true, has_avatar: false };
const MOCK_SENTENCE_1 = "הכול טוב, תודה ששאלת.";
const MOCK_SENTENCE_2 = "איך עבר עליך היום?";
const MOCK_REPLY = `${MOCK_SENTENCE_1} ${MOCK_SENTENCE_2}`;
const CORS = {
  "access-control-allow-origin": "*",
  "access-control-allow-headers": "*",
  "access-control-allow-methods": "GET,POST,OPTIONS",
};

async function mockApi(context, toneMp3, state) {
  let speakCalls = 0;
  const json = (route, status, body) =>
    route.fulfill({ status, headers: CORS, contentType: "application/json", body: JSON.stringify(body) });
  const ndjson = (route, events) =>
    route.fulfill({
      status: 200,
      headers: CORS,
      contentType: "application/x-ndjson",
      body: events.map((e) => JSON.stringify(e)).join("\n") + "\n",
    });

  await context.route(`${API_URL}/**`, async (route) => {
    const req = route.request();
    const { pathname } = new URL(req.url());
    if (req.method() === "OPTIONS") return route.fulfill({ status: 204, headers: CORS });
    if (pathname.startsWith("/api/chat/audio/")) {
      return route.fulfill({ status: 200, headers: { ...CORS, "content-type": "audio/mpeg" }, body: toneMp3 });
    }
    if (req.headers()["x-access-key"] !== MOCK_KEY) return json(route, 401, { detail: "Invalid or missing access key" });

    switch (pathname) {
      case "/api/auth/session":
        return json(route, 200, { role: "family" });
      case "/api/person/list":
        return json(route, 200, [MOCK_PERSON]);
      case "/api/chat/voice-info":
        return json(route, 200, { provider: "elevenlabs", external: true, available: true, chars_used: 120, chars_limit: 20000 });
      case "/api/chat/stream":
        return ndjson(route, [
          { type: "meta", session_id: "s-mock", rag_ms: 120, citations: [], safety_notes: [] },
          { type: "sentence", index: 0, text: MOCK_SENTENCE_1, llm_ms: 350 },
          { type: "sentence", index: 1, text: MOCK_SENTENCE_2, llm_ms: 620 },
          { type: "done", session_id: "s-mock", reply: MOCK_REPLY, rag_ms: 120, llm_first_sentence_ms: 350, llm_total_ms: 620, sentence_count: 2 },
        ]);
      case "/api/chat/text":
        return json(route, 200, { reply: MOCK_REPLY, session_id: "s-mock", citations: [], safety_notes: [] });
      case "/api/chat/speak":
        speakCalls += 1;
        return speakCalls <= 4
          ? json(route, 200, { audio_url: `/api/chat/audio/00000000-0000-0000-0000-00000000000${speakCalls}.mp3`, tts_ms: 280 })
          : json(route, 503, { detail: "budget" });
      case "/api/chat/voice-stream":
        return ndjson(route, [
          { type: "transcription", transcription: TEXT_QUESTION, language: "he", stt_ms: 210, session_id: "s-mock" },
          { type: "meta", session_id: "s-mock", rag_ms: 110, citations: [], safety_notes: [] },
          { type: "sentence", index: 0, text: MOCK_SENTENCE_1, llm_ms: 310 },
          { type: "sentence", index: 1, text: MOCK_SENTENCE_2, llm_ms: 580 },
          { type: "done", session_id: "s-mock", reply: MOCK_REPLY, rag_ms: 110, llm_first_sentence_ms: 310, llm_total_ms: 580, sentence_count: 2 },
        ]);
      case "/api/chat/spoken": {
        const body = JSON.parse(req.postData() ?? "{}");
        state.spokenCommits.push(body);
        return json(route, 200, { session_id: body.session_id ?? "s-mock", committed_chars: (body.spoken_text ?? "").length });
      }
      case "/api/chat/latency": {
        const body = JSON.parse(req.postData() ?? "{}");
        state.latencyReports.push(body);
        return json(route, 200, { status: "ok" });
      }
      default:
        return json(route, 404, { detail: "Not mocked" });
    }
  });
}

// ---------- run ----------

async function main() {
  assert.ok(["mock", "live"].includes(MODE), `E2E_MODE must be mock or live, got ${MODE}`);
  assert.ok(QUESTION_WAV && fs.existsSync(QUESTION_WAV), "Set E2E_QUESTION_WAV to a 16-bit PCM WAV question");
  fs.mkdirSync(OUT_DIR, { recursive: true, mode: 0o700 });

  const browser = await chromium.launch({
    headless: process.env.E2E_HEADED !== "1",
    args: [
      "--use-fake-ui-for-media-stream",
      "--use-fake-device-for-media-stream",
      `--use-file-for-fake-audio-capture=${QUESTION_WAV}`,
      "--autoplay-policy=no-user-gesture-required",
    ],
  });
  const context = await browser.newContext({ locale: "he-IL", viewport: { width: 1280, height: 860 }, permissions: ["microphone"] });
  const mockState = { spokenCommits: [], latencyReports: [] };
  if (MODE === "mock") {
    const tone = process.env.E2E_MOCK_MP3;
    assert.ok(tone && fs.existsSync(tone), "Mock mode needs E2E_MOCK_MP3 (any short MP3)");
    await mockApi(context, fs.readFileSync(tone), mockState);
  }

  const page = await context.newPage();
  const chat = new ChatPage(page);
  const results = {};
  const shots = [];
  const check = (name, ok, extra = {}) => {
    results[name] = { ok: Boolean(ok), ...extra };
  };

  try {
    await chat.open();
    if (await chat.accessKey.isVisible()) shots.push(await chat.shot("1-login"));
    await chat.signIn(MODE === "mock" ? MOCK_KEY : (process.env.E2E_ACCESS_KEY ?? readEnvFileKey("FAMILY_ACCESS_KEY")));

    check("ai_notice_visible", await chat.notice.isVisible());
    const label = (await chat.voiceLabel.textContent()) ?? "";
    check("external_voice_label", label.includes("ElevenLabs") && label.includes("שירות חיצוני"));
    shots.push(await chat.shot("2-empty"));
    const usageBefore = MODE === "live" ? await voiceUsage(page) : null;

    let t = Date.now();
    const textReply = await chat.ask(TEXT_QUESTION, TIMEOUTS.reply);
    const textBody = (await textReply.locator(".body").textContent()) ?? "";
    check("text_reply", textBody.length > 0 && HEBREW.test(textBody) && !(await textReply.evaluate((el) => el.classList.contains("hold"))), {
      seconds: (Date.now() - t) / 1000,
      chars: textBody.length,
    });

    t = Date.now();
    await chat.speak(textReply, TIMEOUTS.speak);
    results.speak_playback = { ...(await checkAudio(page, textReply)), seconds_to_audio: (Date.now() - t) / 1000 };

    t = Date.now();
    const voiceReply = await chat.askByVoice(RECORD_SECONDS, TIMEOUTS.voice);
    const transcript = (await chat.userMessages.last().textContent()) ?? "";
    const voiceBody = (await voiceReply.locator(".body").textContent()) ?? "";
    check("voice_transcription", HEBREW.test(transcript) && (!EXPECTED_TRANSCRIPT || transcript.includes(EXPECTED_TRANSCRIPT)), {
      transcript,
    });
    check("voice_reply", voiceBody.length > 0 && HEBREW.test(voiceBody), { seconds: (Date.now() - t) / 1000, chars: voiceBody.length });
    await voiceReply.getByTestId("reply-audio").waitFor({ timeout: TIMEOUTS.speak });
    results.voice_playback = await checkAudio(page, voiceReply);
    const latText = (await voiceReply.getByTestId("latency-metrics").textContent()) ?? "";
    check("latency_metrics_shown", latText.includes("השהיה לקול:"));
    shots.push(await chat.shot("3-conversation"));

    if (MODE === "mock") {
      // Barge-in while sentence 0 of voiceReply is still playing (2s tone):
      // asking a new question must pause playback, truncate voiceReply to sentence 0,
      // and commit only sentence 0 to /api/chat/spoken.
      const third = await chat.ask("ועוד שאלה", TIMEOUTS.reply);
      const truncatedBody = (await voiceReply.locator(".body").textContent()) ?? "";
      const bargeNoteVisible = await voiceReply.getByTestId("barge-in-note").isVisible();
      const lastCommit = mockState.spokenCommits.at(-1);
      check("barge_in_commits_only_spoken", bargeNoteVisible && truncatedBody === MOCK_SENTENCE_1 && lastCommit?.spoken_text === MOCK_SENTENCE_1, {
        truncatedBody,
        committed: lastCommit?.spoken_text ?? null,
      });

      await chat.speak(third, TIMEOUTS.speak);
      const note = (await third.getByTestId("voice-unavailable").textContent()) ?? "";
      check("budget_message_shown", note.includes("מכסת"));
      await page.setViewportSize({ width: 390, height: 844 });
      shots.push(await chat.shot("4-mobile"));
    } else {
      const usageAfter = await voiceUsage(page);
      results.elevenlabs_chars_this_run = usageBefore && usageAfter ? usageAfter.chars_used - usageBefore.chars_used : null;
    }
  } catch (err) {
    results.error = { ok: false, message: String(err.message ?? err).split("\n")[0] };
    if (process.env.E2E_DEBUG === "1") console.error(err.stack);
    shots.push(await chat.shot("error").catch(() => "screenshot failed"));
  } finally {
    await browser.close();
  }

  const failed = Object.entries(results).filter(([, v]) => v && typeof v === "object" && v.ok === false);
  console.log(JSON.stringify({ mode: MODE, passed: failed.length === 0, results, screenshots: shots }, null, 2));
  process.exitCode = failed.length === 0 ? 0 : 1;
}

main();
