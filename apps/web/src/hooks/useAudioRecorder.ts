"use client";

import { useCallback, useRef, useState } from "react";

interface AudioRecorderState {
  isRecording: boolean;
  duration: number;
  start: () => Promise<void>;
  stop: () => Promise<Blob | null>;
}

export function useAudioRecorder(): AudioRecorderState {
  const [isRecording, setIsRecording] = useState(false);
  const [duration, setDuration] = useState(0);
  const mediaRecorder = useRef<MediaRecorder | null>(null);
  const chunks = useRef<Blob[]>([]);
  const timer = useRef<ReturnType<typeof setInterval> | null>(null);

  const start = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

      // Prefer webm (Chrome/Edge) then mp4 (Safari) then fallback
      const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? "audio/webm;codecs=opus"
        : MediaRecorder.isTypeSupported("audio/mp4")
          ? "audio/mp4"
          : "";

      const recorder = mimeType
        ? new MediaRecorder(stream, { mimeType })
        : new MediaRecorder(stream);

      chunks.current = [];

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunks.current.push(e.data);
        }
      };

      mediaRecorder.current = recorder;
      recorder.start(100); // collect data every 100ms
      setIsRecording(true);
      setDuration(0);

      timer.current = setInterval(() => {
        setDuration((d) => d + 1);
      }, 1000);
    } catch (err) {
      console.error("Microphone access denied:", err);
    }
  }, []);

  const stop = useCallback(async (): Promise<Blob | null> => {
    return new Promise((resolve) => {
      const recorder = mediaRecorder.current;
      if (!recorder || recorder.state === "inactive") {
        resolve(null);
        return;
      }

      if (timer.current) {
        clearInterval(timer.current);
        timer.current = null;
      }

      recorder.onstop = () => {
        const blob = new Blob(chunks.current, { type: recorder.mimeType });
        chunks.current = [];
        setIsRecording(false);
        setDuration(0);

        // Stop all tracks to release mic
        recorder.stream.getTracks().forEach((t) => t.stop());

        resolve(blob);
      };

      recorder.stop();
    });
  }, []);

  return { isRecording, duration, start, stop };
}
