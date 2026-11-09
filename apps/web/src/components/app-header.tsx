"use client";

import { useAccess } from "@/components/access-gate";

export function AppHeader({ title, back }: { title: string; back?: { href: string; label: string } }) {
  const { role, signOut } = useAccess();

  return (
    <header className="mast">
      <h1 className="mark">
        {title}
        <small>סימולציית זיכרון</small>
      </h1>
      <nav>
        {back ? (
          <a href={back.href}>{back.label}</a>
        ) : (
          role === "admin" && <a href="/admin">ניהול</a>
        )}
        <button onClick={signOut} className="btn-pill" data-testid="sign-out">
          יציאה
        </button>
      </nav>
    </header>
  );
}
