import type { Message } from "@/types";

export function exportSessionAsMarkdown(
  ticker: string,
  createdAt: string,
  model: string | null,
  messages: Message[]
): void {
  const date = new Date(createdAt).toLocaleDateString("en-US", {
    year: "numeric",
    month: "long",
    day: "numeric",
  });

  const header = [
    `# ${ticker} — Analyst Agent Session`,
    ``,
    `**Date**: ${date}`,
    model ? `**Model**: ${model}` : "",
    ``,
    `---`,
    ``,
  ].filter(Boolean);

  const body = messages.flatMap((m) => [
    `**${m.role === "user" ? "You" : "Analyst"}**`,
    ``,
    m.content,
    ``,
  ]);

  const md = [...header, ...body].join("\n");
  const filename = `${ticker}-${new Date(createdAt).toISOString().slice(0, 10)}.md`;

  const blob = new Blob([md], { type: "text/markdown" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
