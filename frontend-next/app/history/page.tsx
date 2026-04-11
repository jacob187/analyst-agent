import { ChatHistory } from "@/components/chat/ChatHistory";

export default function HistoryPage() {
  return (
    <div className="mx-auto max-w-2xl px-4 py-12 sm:px-6">
      <div className="mb-8">
        <h1 className="font-display text-2xl font-bold sm:text-3xl">
          Session History
        </h1>
        <p className="mt-2 text-sm text-muted-foreground">
          All past analysis sessions, grouped by ticker.
        </p>
      </div>
      <ChatHistory />
    </div>
  );
}
