"""Conversation memory: token estimation, history compression, and context reconstruction.

These helpers manage the LLM's sliding context window. Full message history
is always preserved in the database — these functions only control what the
LLM sees during a session.
"""

import asyncio

from langchain_core.messages import HumanMessage, AIMessage

from api.db import update_session_summary
from agents.prompts import CONVERSATION_COMPRESSION_PROMPT

# How many tokens (rough estimate) before we compress.
TOKEN_THRESHOLD = 30_000

# After compression, keep this many recent messages verbatim.
RECENT_MESSAGES_TO_KEEP = 6  # 3 user+assistant turns


def estimate_tokens(messages: list) -> int:
    """Rough token count using the ~4 chars/token heuristic.

    Intentionally approximate — just decides when to trigger compression.
    """
    return sum(len(str(m.content)) for m in messages) // 4


def estimate_tokens_incremental(
    history: list, prev_chars: int, prev_len: int
) -> tuple[int, int, int]:
    """Add only new messages to a running token tally.

    Returns ``(tokens, chars, new_len)``:
      - ``tokens`` is the same approximation as ``estimate_tokens(history)``.
      - ``chars`` is the running character total to thread back into the next
        call (tokens are derived as ``chars // 4``; per-message ``// 4``
        accumulation drifts due to truncation, so we track chars).
      - ``new_len`` is ``len(history)`` after this call.

    Callers persist ``(chars, new_len)`` between turns so each subsequent
    call walks only the tail (``history[prev_len:]``). This turns the
    per-turn cost from O(N) to O(new) and the cumulative session cost from
    O(N²) to O(N).

    Pure function — no global state. Callers MUST reset to ``(0, 0)`` when
    ``compress_history`` (or any other call) replaces ``history`` with a
    shorter list, because the previous tally no longer corresponds to a
    prefix of the new list. The function defensively re-baselines if
    ``prev_len > len(history)`` so a forgotten reset still returns a
    correct (but expensive) answer.
    """
    if prev_len > len(history):
        total_chars = sum(len(str(m.content)) for m in history)
        return total_chars // 4, total_chars, len(history)
    added_chars = sum(len(str(m.content)) for m in history[prev_len:])
    chars = prev_chars + added_chars
    return chars // 4, chars, len(history)


async def compress_history(
    session_id: str,
    conversation_history: list,
    llm,
) -> list:
    """Summarize older messages with the LLM, returning a shorter context list.

    Returns [summary_message] + last RECENT_MESSAGES_TO_KEEP messages.
    Full message history is untouched in the database.
    """
    recent = conversation_history[-RECENT_MESSAGES_TO_KEEP:]
    to_summarize = conversation_history[:-RECENT_MESSAGES_TO_KEEP]

    history_lines = []
    for msg in to_summarize:
        content = str(msg.content)
        if content.startswith("[CONVERSATION SUMMARY]"):
            history_lines.append(
                "PRIOR SUMMARY:\n" + content.replace("[CONVERSATION SUMMARY]", "").strip()
            )
        elif msg.type == "human":
            history_lines.append(f"USER: {content}")
        else:
            history_lines.append(f"ASSISTANT: {content}")

    prompt = CONVERSATION_COMPRESSION_PROMPT.format(
        transcript="\n\n".join(history_lines)
    )
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    summary_text = str(response.content)

    asyncio.create_task(update_session_summary(session_id, summary_text))

    summary_message = HumanMessage(content=f"[CONVERSATION SUMMARY]\n{summary_text}")
    return [summary_message] + recent


def build_context_from_session(
    session: dict,
    all_messages: list[dict],
) -> list:
    """Reconstruct conversation_history from DB state.

    If a summary exists and there are more messages than RECENT_MESSAGES_TO_KEEP,
    uses [summary_message] + last N raw messages. Otherwise loads everything.
    """
    def to_lc(msg: dict):
        return (
            HumanMessage(content=msg["content"])
            if msg["role"] == "user"
            else AIMessage(content=msg["content"])
        )

    summary = session.get("summary") if session else None

    if summary and len(all_messages) > RECENT_MESSAGES_TO_KEEP:
        recent_raw = all_messages[-RECENT_MESSAGES_TO_KEEP:]
        return [
            HumanMessage(content=f"[CONVERSATION SUMMARY]\n{summary}")
        ] + [to_lc(m) for m in recent_raw]

    return [to_lc(m) for m in all_messages]
