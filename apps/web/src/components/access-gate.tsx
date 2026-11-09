"use client";

import { createContext, useCallback, useContext, useEffect, useState } from "react";
import { checkAccessKey, clearAccessKey, getAccessKey, setAccessKey, type Role } from "@/lib/api";

interface AccessContextValue {
  role: Role;
  signOut: () => void;
}

const AccessContext = createContext<AccessContextValue | null>(null);

export function useAccess(): AccessContextValue {
  const ctx = useContext(AccessContext);
  if (!ctx) throw new Error("useAccess must be used inside <AccessGate>");
  return ctx;
}

export function AccessGate({ children }: { children: React.ReactNode }) {
  const [role, setRole] = useState<Role | null>(null);
  const [checking, setChecking] = useState(true);
  const [input, setInput] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    checkAccessKey(getAccessKey())
      .then(setRole)
      .catch(() => setError("אין חיבור לשרת"))
      .finally(() => setChecking(false));
  }, []);

  const submit = useCallback(async () => {
    setError("");
    const granted = await checkAccessKey(input.trim()).catch(() => null);
    if (!granted) {
      setError("סיסמה שגויה");
      return;
    }
    setAccessKey(input.trim());
    setRole(granted);
  }, [input]);

  const signOut = useCallback(() => {
    clearAccessKey();
    setRole(null);
    setInput("");
  }, []);

  if (checking) return null;

  if (!role) {
    return (
      <div className="flex-1 flex items-center justify-center px-4 py-10">
        <form
          className="ticket w-full max-w-sm"
          onSubmit={(e) => {
            e.preventDefault();
            submit();
          }}
        >
          <div className="ticket-head">
            <span>זיכרון חי</span>
            <span>למשפחה בלבד</span>
          </div>
          <label htmlFor="access-key" className="block font-medium mb-2">
            סיסמה משפחתית
          </label>
          <div className="flex gap-2">
            <input
              id="access-key"
              type="password"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              autoFocus
              className="field"
              data-testid="access-key"
            />
            <button type="submit" disabled={!input.trim()} className="btn" data-testid="sign-in">
              כניסה
            </button>
          </div>
          {error && (
            <p className="status hold mt-3" role="alert">
              {error}
            </p>
          )}
          <p className="status mt-4">
            סימולציה שנוצרה בבינה מלאכותית מתוך הודעות והקלטות. זה לא האדם עצמו.
          </p>
        </form>
      </div>
    );
  }

  return <AccessContext.Provider value={{ role, signOut }}>{children}</AccessContext.Provider>;
}
