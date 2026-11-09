"use client";

import { useCallback, useEffect, useState } from "react";
import { AccessGate, useAccess } from "@/components/access-gate";
import { AppHeader } from "@/components/app-header";
import { Avatar, usePersonAvatar } from "@/components/person-avatar";
import { apiFetch, UnauthorizedError } from "@/lib/api";

interface Person {
  id: string;
  display_name: string;
  description: string;
  chunk_count: number;
  has_persona: boolean;
  has_avatar: boolean;
}

interface IngestResult {
  status: string;
  title: string;
  chunks: number;
}

interface WhatsAppSender {
  sender: string;
  messages: number;
}

interface WhatsAppIngestResponse {
  status: string;
  sender: string;
  style_examples: number;
  conversation_windows: number;
}

const inputClass = "field text-[15px]";
const buttonClass = "btn shrink-0";

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="border-t border-line pt-5">
      <h2 className="text-xl font-semibold tracking-[-0.02em] mb-4">{title}</h2>
      {children}
    </section>
  );
}

function AdminPanel() {
  const { role, signOut } = useAccess();
  const [persons, setPersons] = useState<Person[]>([]);
  const [selected, setSelected] = useState<Person | null>(null);
  const [newName, setNewName] = useState("");
  const [newDesc, setNewDesc] = useState("");
  const [memoryText, setMemoryText] = useState("");
  const [contributor, setContributor] = useState("");
  const [busy, setBusy] = useState(false);
  const [results, setResults] = useState<IngestResult[]>([]);
  const [waFile, setWaFile] = useState<File | null>(null);
  const [waSenders, setWaSenders] = useState<WhatsAppSender[]>([]);
  const [waSender, setWaSender] = useState("");
  const [status, setStatus] = useState("");
  const [avatarVersion, setAvatarVersion] = useState(0);
  const avatarUrl = usePersonAvatar(selected?.id ?? "", selected?.has_avatar ?? false, avatarVersion);

  const run = useCallback(
    async (label: string, action: () => Promise<string>) => {
      setBusy(true);
      setStatus(label);
      try {
        setStatus(await action());
      } catch (err) {
        if (err instanceof UnauthorizedError) signOut();
        setStatus("הפעולה נכשלה");
      } finally {
        setBusy(false);
      }
    },
    [signOut],
  );

  const loadPersons = useCallback(async () => {
    const data = await apiFetch<Person[]>("/api/person/list");
    setPersons(data);
    setSelected((prev) => data.find((p) => p.id === prev?.id) ?? data[0] ?? prev);
  }, []);

  useEffect(() => {
    loadPersons().catch(() => setStatus("אין חיבור לשרת"));
  }, [loadPersons]);

  if (role !== "admin") {
    return <p className="p-10 text-center text-dim">הדף הזה דורש סיסמת ניהול.</p>;
  }

  const createPerson = () =>
    run("יוצר...", async () => {
      const person = await apiFetch<Person>("/api/person/create", {
        method: "POST",
        body: JSON.stringify({ display_name: newName, description: newDesc }),
      });
      setSelected(person);
      setNewName("");
      setNewDesc("");
      await loadPersons();
      return `נוצר: ${person.display_name}`;
    });

  const detectWhatsAppSenders = (file: File) =>
    run("מזהה משתתפים בקובץ הוואטסאפ...", async () => {
      setWaFile(file);
      const form = new FormData();
      form.append("file", file);
      const senders = await apiFetch<WhatsAppSender[]>("/api/ingest/whatsapp-senders", {
        method: "POST",
        body: form,
      });
      setWaSenders(senders);
      const match =
        senders.find((s) => selected && s.sender.includes(selected.display_name)) ?? senders[0];
      setWaSender(match?.sender ?? "");
      return senders.length > 0
        ? `נמצאו ${senders.length} משתתפים בקובץ. בחרו את האדם המונצח ולחצו על ייבוא.`
        : "לא נמצאו הודעות וואטסאפ בקובץ זה";
    });

  const ingestWhatsApp = () =>
    run("בונה מאגר סגנון ושיחות מוואטסאפ...", async () => {
      if (!waFile || !selected || !waSender) return "חסר קובץ או שם שולח";
      const form = new FormData();
      form.append("file", waFile);
      form.append("person_id", selected.id);
      form.append("sender", waSender);
      const res = await apiFetch<WhatsAppIngestResponse>("/api/ingest/whatsapp", {
        method: "POST",
        body: form,
      });
      await loadPersons();
      return `ייבוא וואטסאפ הושלם (${res.sender}): ${res.style_examples} דוגמאות סגנון ו-${res.conversation_windows} חלונות שיחה נוספו`;
    });

  const uploadFiles = (files: File[]) =>
    run(`מעלה ${files.length} קבצים...`, async () => {
      const out: IngestResult[] = [];
      for (const file of files) {
        const form = new FormData();
        form.append("file", file);
        form.append("person_id", selected!.id);
        form.append("title", file.name);
        out.push(
          await apiFetch<IngestResult>("/api/ingest/document", { method: "POST", body: form }).catch(() => ({
            status: "error",
            title: file.name,
            chunks: 0,
          })),
        );
      }
      setResults(out);
      await loadPersons();
      return `הסתיים: ${out.reduce((s, r) => s + r.chunks, 0)} קטעים נוספו`;
    });

  const uploadAvatar = (file: File) =>
    run("מעלה תמונה...", async () => {
      const form = new FormData();
      form.append("file", file);
      await apiFetch(`/api/person/${selected!.id}/avatar`, { method: "POST", body: form });
      await loadPersons();
      setAvatarVersion((v) => v + 1);
      return "התמונה עודכנה";
    });

  const addMemory = () =>
    run("שומר...", async () => {
      await apiFetch("/api/memory/add", {
        method: "POST",
        body: JSON.stringify({ person_id: selected!.id, text: memoryText, contributor }),
      });
      setMemoryText("");
      return "הזיכרון נשמר";
    });

  const draftPersona = (activate: boolean) =>
    run(activate ? "מייצר ומפעיל פרופיל אישיות..." : "מייצר טיוטה...", async () => {
      const data = await apiFetch<{ status: string }>("/api/persona/draft", {
        method: "POST",
        body: JSON.stringify({
          person_id: selected!.id,
          display_name: selected!.display_name,
          activate,
        }),
      });
      await loadPersons();
      if (data.status !== "ok") return "אין מספיק חומר ליצירת פרופיל";
      return activate
        ? "פרופיל האישיות נוצר והופעל — אפשר להתחיל לשוחח!"
        : "טיוטה נשמרה בקובץ ‎.generated.json. הפרופיל הפעיל לא השתנה.";
    });

  return (
    <div className="max-w-xl w-full mx-auto px-4 py-10 space-y-10">
      <Section title="אדם">
        {persons.length > 0 && (
          <div className="space-y-2 mb-4">
            {persons.map((p) => (
              <button
                key={p.id}
                onClick={() => setSelected(p)}
                aria-pressed={selected?.id === p.id}
                className={`w-full text-right px-4 py-3 rounded-[4px] border border-ink ${
                  selected?.id === p.id ? "bg-card shadow-[-5px_5px_0_var(--yellow)]" : "bg-transparent hover:bg-warm"
                }`}
              >
                <div className="font-medium text-[15px]">{p.display_name}</div>
                <div className="text-xs text-dim mt-0.5">
                  {p.chunk_count} קטעים{p.has_persona ? " · פרופיל פעיל" : ""}
                </div>
              </button>
            ))}
          </div>
        )}
        <div className="flex gap-2">
          <input value={newName} onChange={(e) => setNewName(e.target.value)} placeholder="שם האדם המונצח" className={`flex-1 ${inputClass}`} />
          <input value={newDesc} onChange={(e) => setNewDesc(e.target.value)} placeholder="תיאור קצר" className={`flex-1 ${inputClass}`} />
          <button onClick={createPerson} disabled={busy || !newName.trim()} className={buttonClass}>
            צור
          </button>
        </div>
      </Section>

      {selected && (
        <>
          <Section title="תמונה">
            <label className="flex items-center gap-4 cursor-pointer">
              <Avatar src={avatarUrl} name={selected.display_name} size={72} />
              <span className="text-sm underline underline-offset-4">
                {selected.has_avatar ? "החלפת תמונה" : "העלאת תמונה"}
                <span className="block text-xs text-dim no-underline mt-0.5">‎JPG, PNG או WebP, עד 10MB</span>
              </span>
              <input
                type="file"
                accept="image/jpeg,image/png,image/webp"
                className="hidden"
                disabled={busy}
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file) uploadAvatar(file);
                  e.target.value = "";
                }}
              />
            </label>
          </Section>

          <Section title="ייבוא שיחת וואטסאפ (_chat.txt)">
            <p className="text-sm text-dim mb-3">
              העלו ייצוא שיחה מוואטסאפ (קבוצתית או פרטית). המערכת תזהה את כל השולחים ותבנה אוטומטית את מאגר הסגנון וחלונות השיחה של {selected.display_name}.
            </p>
            <label className="flex flex-col items-center justify-center w-full h-24 border border-dashed border-ink rounded-[4px] bg-card cursor-pointer hover:bg-warm">
              <p className="text-sm m-0">{waFile ? `קובץ נבחר: ${waFile.name}` : "לחצו לבחירת קובץ _chat.txt מוואטסאפ"}</p>
              <input
                type="file"
                accept=".txt"
                className="hidden"
                disabled={busy}
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file) detectWhatsAppSenders(file);
                  e.target.value = "";
                }}
              />
            </label>
            {waSenders.length > 0 && (
              <div className="mt-3 flex gap-2 items-center">
                <select
                  value={waSender}
                  onChange={(e) => setWaSender(e.target.value)}
                  className={`flex-1 ${inputClass}`}
                  aria-label="בחירת שולח בוואטסאפ"
                >
                  {waSenders.map((s) => (
                    <option key={s.sender} value={s.sender}>
                      {s.sender} ({s.messages} הודעות)
                    </option>
                  ))}
                </select>
                <button onClick={ingestWhatsApp} disabled={busy || !waSender} className={buttonClass}>
                  בנה סגנון ושיחות
                </button>
              </div>
            )}
          </Section>

          <Section title="העלאת מסמכים וסיפורים נוספים">
            <label className="flex flex-col items-center justify-center w-full h-28 border border-dashed border-ink rounded-[4px] bg-card cursor-pointer hover:bg-warm">
              <p className="text-sm m-0">{busy ? "מעבד..." : "לחצו להעלאת קבצים"}</p>
              <p className="text-xs text-dim mt-1 mb-0">‎.txt .md .pdf .docx</p>
              <input
                type="file"
                multiple
                accept=".txt,.md,.pdf,.docx,.doc,.eml"
                className="hidden"
                disabled={busy}
                onChange={(e) => {
                  if (e.target.files) uploadFiles(Array.from(e.target.files));
                  e.target.value = "";
                }}
              />
            </label>
            {results.length > 0 && (
              <div className="mt-3 space-y-1">
                {results.map((r, i) => (
                  <div
                    key={i}
                    className={`msg text-xs py-1.5 px-3 max-w-none ${r.status === "ok" ? "ok" : "hold"}`}
                  >
                    {r.title}: {r.chunks} קטעים
                  </div>
                ))}
              </div>
            )}
          </Section>

          <Section title="הוספת זיכרון">
            <div className="space-y-2">
              <textarea
                value={memoryText}
                onChange={(e) => setMemoryText(e.target.value)}
                rows={3}
                placeholder={`סיפור, עובדה או זיכרון על ${selected.display_name}`}
                className={`w-full ${inputClass}`}
              />
              <div className="flex gap-2">
                <input
                  value={contributor}
                  onChange={(e) => setContributor(e.target.value)}
                  placeholder="מי מספר/ת"
                  className={`flex-1 ${inputClass}`}
                />
                <button onClick={addMemory} disabled={busy || !memoryText.trim()} className={buttonClass}>
                  שמור
                </button>
              </div>
            </div>
          </Section>

          <Section title="בניית פרופיל אישיות וסגנון">
            <p className="text-sm text-dim mb-3">
              מנתח את ההודעות והמסמכים של {selected.display_name} ובונה פרופיל סגנון, ביטויים נפוצים ודפוסי שיחה.
            </p>
            <div className="flex flex-wrap gap-2">
              <button onClick={() => draftPersona(true)} disabled={busy || selected.chunk_count === 0} className={buttonClass}>
                צור והפעל פרופיל אישיות
              </button>
              <button onClick={() => draftPersona(false)} disabled={busy || selected.chunk_count === 0} className="btn btn-quiet">
                צור טיוטה בלבד (.generated.json)
              </button>
            </div>
          </Section>
        </>
      )}

      {status && (
        <div className="notice" role="status">
          {status}
        </div>
      )}
    </div>
  );
}

export default function AdminPage() {
  return (
    <div className="min-h-screen flex flex-col">
      <AccessGate>
        <AppHeader title="ניהול" back={{ href: "/", label: "חזרה לצ'אט" }} />
        <AdminPanel />
      </AccessGate>
    </div>
  );
}
