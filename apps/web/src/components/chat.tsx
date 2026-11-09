"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useAccess } from "@/components/access-gate";
import { Avatar, usePersonAvatar } from "@/components/person-avatar";
import { useAudioRecorder } from "@/hooks/useAudioRecorder";
import { ApiError, apiFetch, apiStream, audioUrl, UnauthorizedError } from "@/lib/api";

type VoiceUnavailable = "off" | "blocked" | "budget" | "failed";

export interface TurnLatency {
  mode: "voice" | "text";
  stt_ms?: number;
  rag_ms?: number;
  llm_first_sentence_ms?: number;
  llm_total_ms?: number;
  tts_first_audio_ms?: number;
  client_e2e_ms?: number;
}

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  userQuestion?: string;
  sentences?: string[];
  audioUrl?: string;
  autoPlay?: boolean;
  voiceUnavailable?: VoiceUnavailable;
  interrupted?: boolean;
  latency?: TurnLatency;
  error?: boolean;
}

interface Person {
  id: string;
  display_name: string;
  chunk_count: number;
  has_persona: boolean;
  has_avatar: boolean;
}

interface ChatReply {
  reply: string;
  session_id: string;
}

interface VoiceReply extends ChatReply {
  transcription: string;
  audio_url: string;
  voice_unavailable: VoiceUnavailable | "";
  stt_ms?: number;
  tts_ms?: number;
}

interface SpeakReply {
  audio_url: string;
  tts_ms?: number;
}

interface VoiceInfo {
  provider: string;
  external: boolean;
  available: boolean;
  chars_used: number;
  chars_limit: number;
}

interface ActivePlayback {
  msgId: string;
  sessionId: string | null;
  userQuestion: string;
  mode: "voice" | "text";
  tSpeechEnd?: number;
  sentences: string[];
  audioUrls: (string | undefined)[];
  inFlightSpeak: Set<number>;
  currentIndex: number;
  startedCurrent: boolean;
  completedCount: number;
  streamDone: boolean;
  aborted: boolean;
  latency: TurnLatency;
}

const CONNECTION_ERROR = "אין חיבור לשרת. נסו שוב עוד רגע.";

const VOICE_UNAVAILABLE_TEXT: Record<VoiceUnavailable, string> = {
  off: "הקול כבוי כרגע. התשובה בטקסט בלבד.",
  blocked: "הקול כבוי עד שמאשרים שירות קול חיצוני.",
  budget: "נגמרה מכסת הקול החודשית. התשובה בטקסט בלבד.",
  failed: "שירות הקול לא ענה. התשובה בטקסט בלבד.",
};

function voiceLabel(info: VoiceInfo | null): string {
  if (!info?.available) return "הקול כבוי";
  if (info.provider === "elevenlabs") return "קול סינתטי מ-ElevenLabs, שירות חיצוני: טקסט התשובה נשלח אליהם";
  return "קול סינתטי";
}

function unavailableReason(err: unknown): VoiceUnavailable {
  if (err instanceof ApiError && err.status === 503 && err.detail in VOICE_UNAVAILABLE_TEXT) {
    return err.detail as VoiceUnavailable;
  }
  return "failed";
}

function splitClientSentences(text: string): string[] {
  const parts = text
    .split(/(?<=[.!?…]+[\s\u{1F300}-\u{1FAFF}\u2600-\u27BF]*)\s+|\n+/u)
    .map((s) => s.trim())
    .filter(Boolean);
  return parts.length > 0 ? parts : [text.trim()].filter(Boolean);
}

function formatLatency(lat: TurnLatency): string {
  const sec = (ms?: number) => (ms !== undefined ? `${(ms / 1000).toFixed(1)}s` : null);
  const parts: string[] = [];
  if (lat.stt_ms !== undefined) parts.push(`תמלול ${sec(lat.stt_ms)}`);
  if (lat.rag_ms !== undefined) parts.push(`חיפוש ${sec(lat.rag_ms)}`);
  if (lat.llm_first_sentence_ms !== undefined) parts.push(`משפט ראשון ${sec(lat.llm_first_sentence_ms)}`);
  if (lat.tts_first_audio_ms !== undefined) parts.push(`קול ${sec(lat.tts_first_audio_ms)}`);
  if (lat.client_e2e_ms !== undefined) {
    return `השהיה לקול: ${sec(lat.client_e2e_ms)}${parts.length ? ` (${parts.join(" · ")})` : ""}`;
  }
  return parts.join(" · ");
}

export function Chat() {
  const { role, signOut } = useAccess();
  const [persons, setPersons] = useState<Person[] | null>(null);
  const [personId, setPersonId] = useState<string>("");
  const [personName, setPersonName] = useState<string>("");
  const [hasAvatar, setHasAvatar] = useState(false);
  const avatarUrl = usePersonAvatar(personId, hasAvatar);
  const [voice, setVoice] = useState<VoiceInfo | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [speakingId, setSpeakingId] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const playbackRef = useRef<ActivePlayback | null>(null);
  const audioElsRef = useRef<Map<string, HTMLAudioElement>>(new Map());
  const { isRecording, duration, start: startRecording, stop: stopRecording } = useAudioRecorder();

  const handleError = useCallback(
    (err: unknown) => {
      if (err instanceof UnauthorizedError) signOut();
    },
    [signOut],
  );

  useEffect(() => {
    apiFetch<Person[]>("/api/person/list")
      .then((data) => {
        setPersons(data);
        const ready = data.find((p) => p.has_persona && p.chunk_count > 0) ?? data[0];
        if (ready) {
          setPersonId(ready.id);
          setPersonName(ready.display_name);
          setHasAvatar(ready.has_avatar);
        }
      })
      .catch((err) => {
        handleError(err);
        setPersons([]);
      });
    apiFetch<VoiceInfo>("/api/chat/voice-info")
      .then(setVoice)
      .catch(() => setVoice(null));
  }, [handleError]);

  const scrollToBottom = useCallback(() => {
    setTimeout(() => {
      scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
    }, 50);
  }, []);

  const updateMessage = useCallback(
    (id: string, patch: Partial<Message> | ((prev: Message) => Partial<Message>)) =>
      setMessages((prev) =>
        prev.map((m) => {
          if (m.id !== id) return m;
          const next = typeof patch === "function" ? patch(m) : patch;
          return { ...m, ...next };
        }),
      ),
    [],
  );

  const commitSpokenToHistory = useCallback(
    (sid: string | null, userQuestion: string, spokenText: string) => {
      if (!sid || !userQuestion.trim()) return;
      apiFetch("/api/chat/spoken", {
        method: "POST",
        body: JSON.stringify({
          session_id: sid,
          user_message: userQuestion,
          spoken_text: spokenText,
        }),
      }).catch(() => undefined);
    },
    [],
  );

  const reportLatency = useCallback((lat: TurnLatency, extra: Record<string, unknown> = {}) => {
    apiFetch("/api/chat/latency", {
      method: "POST",
      body: JSON.stringify({ ...lat, ...extra }),
    }).catch(() => undefined);
  }, []);

  /** Barge-in: stop active audio playback, abort in-flight stream/TTS, and commit only what was heard. */
  const stopPlaybackAndStream = useCallback(() => {
    if (abortRef.current) {
      abortRef.current.abort();
      abortRef.current = null;
    }
    for (const el of audioElsRef.current.values()) {
      if (!el.paused) {
        el.pause();
      }
    }
    const pb = playbackRef.current;
    if (!pb || pb.aborted) return;
    pb.aborted = true;
    playbackRef.current = null;
    setSpeakingId(null);

    const totalSentences = pb.sentences.length;
    const heardCount = pb.startedCurrent
      ? Math.max(pb.completedCount, pb.currentIndex + 1)
      : pb.completedCount;
    const allCompleted = pb.streamDone && totalSentences > 0 && pb.completedCount >= totalSentences;

    if (!allCompleted && totalSentences > 0) {
      const keptSentences = pb.sentences.slice(0, heardCount);
      const spokenText = keptSentences.join(" ").trim();
      commitSpokenToHistory(pb.sessionId, pb.userQuestion, spokenText);
      updateMessage(pb.msgId, (prev) => ({
        content: spokenText || prev.content,
        sentences: keptSentences.length > 0 ? keptSentences : prev.sentences,
        interrupted: true,
        autoPlay: false,
      }));
      reportLatency(pb.latency, {
        interrupted: true,
        sentence_count: totalSentences,
        spoken_sentences: heardCount,
        reply_chars: pb.sentences.join(" ").length,
        spoken_chars: spokenText.length,
      });
    }
  }, [commitSpokenToHistory, reportLatency, updateMessage]);

  const selectPerson = (p: Person) => {
    stopPlaybackAndStream();
    setPersonId(p.id);
    setPersonName(p.display_name);
    setHasAvatar(p.has_avatar);
    setMessages([]);
    setSessionId(null);
  };

  const synthesizeSentence = useCallback(
    async (pb: ActivePlayback, index: number, signal: AbortSignal) => {
      if (pb.aborted || pb.audioUrls[index] || pb.inFlightSpeak.has(index)) return;
      const text = pb.sentences[index];
      if (!text) return;

      pb.inFlightSpeak.add(index);
      const started = performance.now();
      try {
        const data = await apiFetch<SpeakReply>("/api/chat/speak", {
          method: "POST",
          body: JSON.stringify({ text }),
          signal,
        });
        if (pb.aborted) return;
        const resolvedUrl = audioUrl(data.audio_url);
        pb.audioUrls[index] = resolvedUrl;
        if (index === 0 && pb.latency.tts_first_audio_ms === undefined) {
          pb.latency.tts_first_audio_ms = Math.round(data.tts_ms ?? performance.now() - started);
        }
        if (pb.currentIndex === index) {
          updateMessage(pb.msgId, {
            audioUrl: resolvedUrl,
            autoPlay: true,
            latency: { ...pb.latency },
          });
        }
      } catch (err) {
        if (pb.aborted || (err instanceof DOMException && err.name === "AbortError")) return;
        handleError(err);
        if (pb.currentIndex === index) {
          updateMessage(pb.msgId, { voiceUnavailable: unavailableReason(err) });
          pb.aborted = true;
          playbackRef.current = null;
          setSpeakingId(null);
        }
      } finally {
        pb.inFlightSpeak.delete(index);
        apiFetch<VoiceInfo>("/api/chat/voice-info")
          .then(setVoice)
          .catch(() => undefined);
      }
    },
    [handleError, updateMessage],
  );

  const handleAudioPlay = useCallback(
    (msgId: string) => {
      const pb = playbackRef.current;
      if (!pb || pb.msgId !== msgId || pb.aborted) return;
      pb.startedCurrent = true;
      if (pb.currentIndex === 0 && pb.tSpeechEnd !== undefined && pb.latency.client_e2e_ms === undefined) {
        pb.latency.client_e2e_ms = Math.round(performance.now() - pb.tSpeechEnd);
        updateMessage(msgId, { latency: { ...pb.latency } });
      }
      const nextIdx = pb.currentIndex + 1;
      if (nextIdx < pb.sentences.length && abortRef.current) {
        void synthesizeSentence(pb, nextIdx, abortRef.current.signal);
      }
    },
    [synthesizeSentence, updateMessage],
  );

  const handleAudioEnded = useCallback(
    (msgId: string) => {
      const pb = playbackRef.current;
      if (!pb || pb.msgId !== msgId || pb.aborted) return;

      pb.completedCount = pb.currentIndex + 1;
      pb.startedCurrent = false;
      const spokenSoFar = pb.sentences.slice(0, pb.completedCount).join(" ").trim();
      commitSpokenToHistory(pb.sessionId, pb.userQuestion, spokenSoFar);

      const nextIdx = pb.currentIndex + 1;
      if (nextIdx < pb.sentences.length) {
        pb.currentIndex = nextIdx;
        const nextUrl = pb.audioUrls[nextIdx];
        if (nextUrl) {
          updateMessage(msgId, { audioUrl: nextUrl, autoPlay: true });
        } else if (abortRef.current) {
          void synthesizeSentence(pb, nextIdx, abortRef.current.signal);
        }
      } else if (pb.streamDone) {
        reportLatency(pb.latency, {
          interrupted: false,
          sentence_count: pb.sentences.length,
          spoken_sentences: pb.completedCount,
          reply_chars: spokenSoFar.length,
          spoken_chars: spokenSoFar.length,
        });
        playbackRef.current = null;
        setSpeakingId(null);
      }
    },
    [commitSpokenToHistory, reportLatency, synthesizeSentence, updateMessage],
  );

  const sendMessage = useCallback(async () => {
    const text = input.trim();
    if (!text || loading || !personId) return;

    stopPlaybackAndStream();
    const userMsgId = crypto.randomUUID();
    const assistantMsgId = crypto.randomUUID();
    setMessages((prev) => [...prev, { id: userMsgId, role: "user", content: text }]);
    setInput("");
    setLoading(true);
    scrollToBottom();

    const controller = new AbortController();
    abortRef.current = controller;
    const latency: TurnLatency = { mode: "text" };
    let createdAssistant = false;
    const collectedSentences: string[] = [];

    try {
      await apiStream(
        "/api/chat/stream",
        {
          method: "POST",
          body: JSON.stringify({
            person_id: personId,
            session_id: sessionId,
            message: text,
            defer_history: false,
          }),
          signal: controller.signal,
        },
        (event) => {
          const type = String(event.type ?? "");
          if (type === "meta") {
            const sid = String(event.session_id ?? "");
            if (sid) setSessionId(sid);
            if (typeof event.rag_ms === "number") latency.rag_ms = Math.round(event.rag_ms);
          } else if (type === "sentence") {
            const sentence = String(event.text ?? "").trim();
            if (!sentence) return;
            collectedSentences.push(sentence);
            if (collectedSentences.length === 1 && typeof event.llm_ms === "number") {
              latency.llm_first_sentence_ms = Math.round(event.llm_ms);
            }
            const contentNow = collectedSentences.join(" ");
            if (!createdAssistant) {
              createdAssistant = true;
              setMessages((prev) => [
                ...prev,
                {
                  id: assistantMsgId,
                  role: "assistant",
                  content: contentNow,
                  userQuestion: text,
                  sentences: [...collectedSentences],
                  latency: { ...latency },
                },
              ]);
            } else {
              updateMessage(assistantMsgId, {
                content: contentNow,
                sentences: [...collectedSentences],
                latency: { ...latency },
              });
            }
            scrollToBottom();
          } else if (type === "done") {
            const sid = String(event.session_id ?? "");
            if (sid) setSessionId(sid);
            if (typeof event.llm_total_ms === "number") {
              latency.llm_total_ms = Math.round(event.llm_total_ms);
            }
            updateMessage(assistantMsgId, { latency: { ...latency } });
            reportLatency(latency, {
              interrupted: false,
              sentence_count: collectedSentences.length,
              reply_chars: collectedSentences.join(" ").length,
            });
          }
        },
      );
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") return;
      if (err instanceof ApiError && err.status === 404) {
        try {
          const data = await apiFetch<ChatReply>("/api/chat/text", {
            method: "POST",
            body: JSON.stringify({ person_id: personId, session_id: sessionId, message: text }),
            signal: controller.signal,
          });
          setSessionId(data.session_id);
          setMessages((prev) => [
            ...prev,
            {
              id: assistantMsgId,
              role: "assistant",
              content: data.reply,
              userQuestion: text,
              sentences: splitClientSentences(data.reply),
            },
          ]);
          return;
        } catch (fallbackErr) {
          handleError(fallbackErr);
        }
      } else {
        handleError(err);
      }
      setMessages((prev) => [
        ...prev,
        { id: assistantMsgId, role: "assistant", content: CONNECTION_ERROR, error: true },
      ]);
    } finally {
      setLoading(false);
      scrollToBottom();
    }
  }, [
    input,
    loading,
    personId,
    sessionId,
    stopPlaybackAndStream,
    scrollToBottom,
    updateMessage,
    reportLatency,
    handleError,
  ]);

  const speak = useCallback(
    async (msg: Message) => {
      if (speakingId) return;
      stopPlaybackAndStream();
      setSpeakingId(msg.id);

      const sentences = msg.sentences?.length ? msg.sentences : splitClientSentences(msg.content);
      const controller = new AbortController();
      abortRef.current = controller;

      const pb: ActivePlayback = {
        msgId: msg.id,
        sessionId,
        userQuestion: msg.userQuestion ?? "",
        mode: "text",
        sentences: [...sentences],
        audioUrls: new Array(sentences.length).fill(undefined),
        inFlightSpeak: new Set(),
        currentIndex: 0,
        startedCurrent: false,
        completedCount: 0,
        streamDone: true,
        aborted: false,
        latency: { ...(msg.latency ?? { mode: "text" }) },
      };
      playbackRef.current = pb;

      await synthesizeSentence(pb, 0, controller.signal);
    },
    [speakingId, sessionId, stopPlaybackAndStream, synthesizeSentence],
  );

  const handleVoiceToggle = useCallback(async () => {
    if (!personId) return;
    if (!isRecording) {
      // Barge-in: starting to speak stops any ongoing playback/stream immediately.
      stopPlaybackAndStream();
      await startRecording();
      return;
    }

    const tSpeechEnd = performance.now();
    const audioBlob = await stopRecording();
    if (!audioBlob || audioBlob.size < 1000) return;

    setLoading(true);
    scrollToBottom();
    const userMsgId = crypto.randomUUID();
    const assistantMsgId = crypto.randomUUID();
    setMessages((prev) => [...prev, { id: userMsgId, role: "user", content: "מתמלל..." }]);

    const controller = new AbortController();
    abortRef.current = controller;
    const latency: TurnLatency = { mode: "voice" };
    const pb: ActivePlayback = {
      msgId: assistantMsgId,
      sessionId,
      userQuestion: "",
      mode: "voice",
      tSpeechEnd,
      sentences: [],
      audioUrls: [],
      inFlightSpeak: new Set(),
      currentIndex: 0,
      startedCurrent: false,
      completedCount: 0,
      streamDone: false,
      aborted: false,
      latency,
    };
    playbackRef.current = pb;

    const formData = new FormData();
    const ext = audioBlob.type.includes("mp4") ? ".mp4" : ".webm";
    formData.append("audio", audioBlob, `recording${ext}`);
    formData.append("person_id", personId);
    if (sessionId) formData.append("session_id", sessionId);

    let createdAssistant = false;

    try {
      await apiStream(
        "/api/chat/voice-stream",
        { method: "POST", body: formData, signal: controller.signal },
        (event) => {
          if (pb.aborted) return;
          const type = String(event.type ?? "");
          if (type === "transcription") {
            const transcript = String(event.transcription ?? "").trim();
            const sid = String(event.session_id ?? "");
            if (sid) {
              setSessionId(sid);
              pb.sessionId = sid;
            }
            if (typeof event.stt_ms === "number") {
              latency.stt_ms = Math.round(event.stt_ms);
            }
            pb.userQuestion = transcript;
            updateMessage(userMsgId, { content: transcript || "(לא נשמע דיבור)" });
          } else if (type === "meta") {
            const sid = String(event.session_id ?? "");
            if (sid) {
              setSessionId(sid);
              pb.sessionId = sid;
            }
            if (typeof event.rag_ms === "number") {
              latency.rag_ms = Math.round(event.rag_ms);
            }
          } else if (type === "sentence") {
            const sentence = String(event.text ?? "").trim();
            if (!sentence) return;
            const idx = pb.sentences.length;
            pb.sentences.push(sentence);
            pb.audioUrls.push(undefined);
            if (idx === 0 && typeof event.llm_ms === "number") {
              latency.llm_first_sentence_ms = Math.round(event.llm_ms);
            }
            const contentNow = pb.sentences.join(" ");
            if (!createdAssistant) {
              createdAssistant = true;
              setMessages((prev) => [
                ...prev,
                {
                  id: assistantMsgId,
                  role: "assistant",
                  content: contentNow,
                  userQuestion: pb.userQuestion,
                  sentences: [...pb.sentences],
                  latency: { ...latency },
                },
              ]);
            } else {
              updateMessage(assistantMsgId, {
                content: contentNow,
                sentences: [...pb.sentences],
                latency: { ...latency },
              });
            }
            scrollToBottom();
            if (idx === pb.currentIndex || (idx === pb.currentIndex + 1 && pb.startedCurrent)) {
              void synthesizeSentence(pb, idx, controller.signal);
            }
          } else if (type === "done") {
            pb.streamDone = true;
            if (typeof event.llm_total_ms === "number") {
              latency.llm_total_ms = Math.round(event.llm_total_ms);
            }
            updateMessage(assistantMsgId, { latency: { ...latency } });
          }
        },
      );
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") return;
      if (err instanceof ApiError && err.status === 404) {
        // Fallback for legacy non-streaming `/api/chat/voice` mocks
        const fallbackForm = new FormData();
        fallbackForm.append("audio", audioBlob, `recording${ext}`);
        fallbackForm.append("person_id", personId);
        if (sessionId) fallbackForm.append("session_id", sessionId);
        try {
          const data = await apiFetch<VoiceReply>("/api/chat/voice", {
            method: "POST",
            body: fallbackForm,
            signal: controller.signal,
          });
          setSessionId(data.session_id);
          pb.sessionId = data.session_id;
          pb.userQuestion = data.transcription;
          pb.sentences = splitClientSentences(data.reply);
          pb.streamDone = true;
          updateMessage(userMsgId, { content: data.transcription || "(לא נשמע דיבור)" });
          const resolvedUrl = data.audio_url ? audioUrl(data.audio_url) : undefined;
          pb.audioUrls = [resolvedUrl];
          setMessages((prev) => [
            ...prev,
            {
              id: assistantMsgId,
              role: "assistant",
              content: data.reply,
              userQuestion: data.transcription,
              sentences: pb.sentences,
              audioUrl: resolvedUrl,
              autoPlay: true,
              voiceUnavailable: data.voice_unavailable || undefined,
              latency: { ...latency },
            },
          ]);
          return;
        } catch (fallbackErr) {
          handleError(fallbackErr);
        }
      } else {
        handleError(err);
      }
      playbackRef.current = null;
      updateMessage(userMsgId, { content: "ההקלטה לא עברה. נסו שוב.", error: true });
    } finally {
      setLoading(false);
      scrollToBottom();
      apiFetch<VoiceInfo>("/api/chat/voice-info")
        .then(setVoice)
        .catch(() => undefined);
    }
  }, [
    personId,
    isRecording,
    stopPlaybackAndStream,
    startRecording,
    stopRecording,
    scrollToBottom,
    sessionId,
    updateMessage,
    synthesizeSentence,
    handleError,
  ]);

  const formatDuration = (s: number) => `${Math.floor(s / 60)}:${(s % 60).toString().padStart(2, "0")}`;

  if (persons === null) return null;

  if (persons.length === 0) {
    return (
      <div className="flex-1 flex items-center justify-center px-4">
        <div className="ticket max-w-sm text-center">
          <p className="mb-4">עדיין לא הוגדר אף אחד.</p>
          {role === "admin" && (
            <a href="/admin" className="btn inline-block no-underline">
              להגדרה
            </a>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col flex-1 overflow-hidden">
      <div className="notice" data-testid="ai-notice">
        סימולציה שנוצרה בבינה מלאכותית מתוך הודעות והקלטות של {personName || "האדם המונצח"}. זה לא {personName || "האדם"} עצמו.{" "}
        <span data-testid="voice-label">{voiceLabel(voice)}.</span>
      </div>

      {persons.length > 1 && (
        <div className="border-b border-line px-4 py-2 flex gap-2 overflow-x-auto">
          {persons.map((p) => (
            <button
              key={p.id}
              onClick={() => selectPerson(p)}
              aria-pressed={personId === p.id}
              className={personId === p.id ? "btn" : "btn btn-quiet"}
            >
              {p.display_name}
            </button>
          ))}
        </div>
      )}

      <div ref={scrollRef} className="flex-1 overflow-y-auto px-4 py-8">
        <div className="max-w-2xl mx-auto space-y-4">
          {messages.length === 0 && (
            <div className="grid gap-8 sm:grid-cols-[minmax(0,200px)_minmax(0,1fr)] items-start py-4">
              <figure className="portrait m-0 justify-self-center sm:justify-self-stretch">
                <div className="flex justify-center">
                  <Avatar src={avatarUrl} name={personName} size={160} />
                </div>
                <figcaption>{personName || "..."}</figcaption>
              </figure>
              <div>
                <h2 className="text-[clamp(34px,5vw,52px)] font-semibold leading-[1.05] tracking-[-0.03em] mb-5">
                  לדבר עם {personName || "..."}
                </h2>
                <ol className="manual">
                  <li>
                    <h3>כותבים או מקליטים</h3>
                    <p>לחיצה על המיקרופון מתחילה הקלטה (ועוצרת הקראה פעילה), לחיצה נוספת שולחת.</p>
                  </li>
                  <li>
                    <h3>התשובה נבנית מהארכיון</h3>
                    <p>מתוך ההודעות שלו. כשאין מידע, התשובה אמורה לומר את זה.</p>
                  </li>
                  <li>
                    <h3>השמעה בקול סינתטי</h3>
                    <p>הקול מחקה את שלו משפט־אחר־משפט. נשמר בהיסטוריה רק מה שהושמע בפועל.</p>
                  </li>
                </ol>
              </div>
            </div>
          )}

          {messages.map((msg) =>
            msg.role === "user" ? (
              <div key={msg.id} className="flex justify-start">
                <div className="msg-user" data-testid="message-user">
                  <p className="m-0 whitespace-pre-wrap">{msg.content}</p>
                </div>
              </div>
            ) : (
              <div key={msg.id} className="flex items-start gap-3 justify-end">
                <div className={`msg ${msg.error ? "hold" : ""}`} data-testid="message-assistant">
                  <div className="hdr">{personName} (סימולציה)</div>
                  <p className="body m-0">{msg.content}</p>
                  {msg.interrupted && (
                    <p className="status mt-1.5 mb-0" data-testid="barge-in-note">
                      ההקראה הופסקה — נשמר בשיחה רק מה שהושמע.
                    </p>
                  )}
                  {msg.latency && formatLatency(msg.latency) && (
                    <p className="status mt-1.5 mb-0" data-testid="latency-metrics">
                      {formatLatency(msg.latency)}
                    </p>
                  )}
                  {!msg.error &&
                    (msg.audioUrl ? (
                      <audio
                        ref={(el) => {
                          if (el) audioElsRef.current.set(msg.id, el);
                          else audioElsRef.current.delete(msg.id);
                        }}
                        controls
                        autoPlay={msg.autoPlay}
                        preload="auto"
                        className="mt-2 w-full h-9"
                        src={msg.audioUrl}
                        onPlay={() => handleAudioPlay(msg.id)}
                        onEnded={() => handleAudioEnded(msg.id)}
                        data-testid="reply-audio"
                      />
                    ) : msg.voiceUnavailable ? (
                      <p className="status mt-2 mb-0" data-testid="voice-unavailable">
                        {VOICE_UNAVAILABLE_TEXT[msg.voiceUnavailable]}
                      </p>
                    ) : (
                      voice?.available && (
                        <button
                          onClick={() => speak(msg)}
                          disabled={speakingId !== null}
                          className="speak"
                          data-testid="speak"
                          title={voiceLabel(voice)}
                        >
                          {speakingId === msg.id ? "מכין קול..." : "השמע"}
                        </button>
                      )
                    ))}
                </div>
                <Avatar src={avatarUrl} name={personName} size={36} />
              </div>
            ),
          )}

          {loading && (
            <div className="flex items-start gap-3 justify-end" data-testid="typing">
              <div className="msg">
                <div className="hdr">{personName} (סימולציה)</div>
                <div className="typing flex gap-1.5 py-1" aria-label="כותב תשובה">
                  <span />
                  <span />
                  <span />
                </div>
              </div>
              <Avatar src={avatarUrl} name={personName} size={36} />
            </div>
          )}
        </div>
      </div>

      <div className="composer">
        <div className="max-w-2xl mx-auto flex gap-2 items-center">
          <button
            onClick={handleVoiceToggle}
            disabled={!personId}
            aria-label={isRecording ? "עצור ושלח הקלטה" : "הקלט"}
            aria-pressed={isRecording}
            className="btn record"
            data-testid="record"
          >
            {isRecording ? (
              <span className="text-xs font-semibold">{formatDuration(duration)}</span>
            ) : (
              <svg className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth={1.8} viewBox="0 0 24 24" aria-hidden>
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 1a4 4 0 0 1 4 4v6a4 4 0 0 1-8 0V5a4 4 0 0 1 4-4z" />
                <path strokeLinecap="round" strokeLinejoin="round" d="M19 10v1a7 7 0 0 1-14 0v-1M12 19v4M8 23h8" />
              </svg>
            )}
          </button>

          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
              }
            }}
            placeholder={isRecording ? "מקליט..." : "כתבו הודעה"}
            aria-label="הודעה"
            className="field"
            disabled={isRecording || !personId}
            data-testid="message-input"
          />

          <button
            onClick={sendMessage}
            disabled={loading || !input.trim() || isRecording || !personId}
            className="btn"
            data-testid="send"
          >
            שליחה
          </button>
        </div>
      </div>
    </div>
  );
}
