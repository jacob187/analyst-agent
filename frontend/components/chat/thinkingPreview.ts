const MAX_LINES = 2;
const MAX_CHARS = 160;

/**
 * Compact tail of a streamed thinking transcript: the last non-empty lines,
 * trimmed from the front to MAX_CHARS. Returns "" when there is nothing to show.
 */
export function latestThinkingPreview(thinking: string | undefined): string {
  if (!thinking) return "";
  const lines = thinking
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .slice(-MAX_LINES);
  const joined = lines.join("\n");
  if (joined.length <= MAX_CHARS) return joined;
  return `…${joined.slice(-(MAX_CHARS - 1)).trimStart()}`;
}
