export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

const KEY_STORAGE = "rony.accessKey";

export type Role = "family" | "admin";

export class UnauthorizedError extends Error {}

export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly detail = "",
  ) {
    super(`API error ${status}`);
  }
}

async function apiError(res: Response): Promise<ApiError> {
  const body = (await res.json().catch(() => null)) as { detail?: unknown } | null;
  return new ApiError(res.status, typeof body?.detail === "string" ? body.detail : "");
}

export function getAccessKey(): string {
  if (typeof window === "undefined") return "";
  return window.localStorage.getItem(KEY_STORAGE) ?? "";
}

export function setAccessKey(key: string): void {
  window.localStorage.setItem(KEY_STORAGE, key);
}

export function clearAccessKey(): void {
  window.localStorage.removeItem(KEY_STORAGE);
}

async function request(path: string, init: RequestInit = {}): Promise<Response> {
  const headers = new Headers(init.headers);
  headers.set("X-Access-Key", getAccessKey());
  if (init.body && !(init.body instanceof FormData)) {
    headers.set("Content-Type", "application/json");
  }

  const res = await fetch(`${API_URL}${path}`, { ...init, headers });
  if (res.status === 401) {
    clearAccessKey();
    throw new UnauthorizedError("Access key rejected");
  }
  return res;
}

export async function apiFetch<T>(path: string, init: RequestInit = {}): Promise<T> {
  const res = await request(path, init);
  if (!res.ok) throw await apiError(res);
  return (await res.json()) as T;
}

/** Stream NDJSON events from a POST/GET endpoint, invoking `onEvent` per line. */
export async function apiStream<T = Record<string, unknown>>(
  path: string,
  init: RequestInit,
  onEvent: (event: T) => void | Promise<void>,
): Promise<void> {
  const res = await request(path, init);
  if (!res.ok) throw await apiError(res);
  if (!res.body) return;

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let newlineIdx = buffer.indexOf("\n");
    while (newlineIdx !== -1) {
      const line = buffer.slice(0, newlineIdx).trim();
      buffer = buffer.slice(newlineIdx + 1);
      if (line) {
        await onEvent(JSON.parse(line) as T);
      }
      newlineIdx = buffer.indexOf("\n");
    }
  }
  const tail = buffer.trim();
  if (tail) {
    await onEvent(JSON.parse(tail) as T);
  }
}

/** Fetch a protected file (e.g. a photo) as a blob; null when it doesn't exist. */
export async function apiFetchBlob(path: string): Promise<Blob | null> {
  const res = await request(path);
  if (res.status === 404) return null;
  if (!res.ok) throw await apiError(res);
  return res.blob();
}

export async function checkAccessKey(key: string): Promise<Role | null> {
  const res = await fetch(`${API_URL}/api/auth/session`, {
    headers: { "X-Access-Key": key },
  });
  if (!res.ok) return null;
  const data = (await res.json()) as { role: Role };
  return data.role;
}

export function audioUrl(path: string): string {
  return `${API_URL}${path}`;
}
