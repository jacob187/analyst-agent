const MAX_LINES = 2;
const MAX_CHARS = 160;

/**
 * Compact tail of a streamed thinking transcript: the last non-empty lines,
 * trimmed from the front to MAX_CHARS. Returns "" when there is nothing to show.
 */
export function latestThinkingPreview(thinking: string | undefined): string {
  if (!thinking) return "";
  // Scan backwards so the cost tracks the preview, not the whole transcript
  // (this runs on every streamed chunk).
  const lines: string[] = [];
  let end = thinking.length;
  while (end > 0 && lines.length < MAX_LINES) {
    const newline = thinking.lastIndexOf("\n", end - 1);
    const line = thinking.slice(newline + 1, end).trim();
    if (line) lines.unshift(line);
    end = newline;
  }
  const joined = lines.join("\n");
  if (joined.length <= MAX_CHARS) return joined;
  return `…${joined.slice(-(MAX_CHARS - 1)).trimStart()}`;
}
