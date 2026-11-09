import { AccessGate } from "@/components/access-gate";
import { AppHeader } from "@/components/app-header";
import { Chat } from "@/components/chat";

export default function Home() {
  return (
    <div className="flex flex-col h-screen">
      <AccessGate>
        <AppHeader title="זיכרון חי" />
        <Chat />
      </AccessGate>
    </div>
  );
}
