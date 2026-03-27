"""Conversation memory: token estimation, history compression, and context reconstruction.

These helpers manage the LLM's sliding context window. Full message history
is always preserved in the database — these functions only control what the
LLM sees during a session.
"""

import asyncio

from langchain_core.messages import HumanMessage, AIMessage

from api.db import update_session_summary

# How many tokens (rough estimate) before we compress.
TOKEN_THRESHOLD = 30_000

# After compression, keep this many recent messages verbatim.
RECENT_MESSAGES_TO_KEEP = 6  # 3 user+assistant turns


def estimate_tokens(messages: list) -> int:
    """Rough token count using the ~4 chars/token heuristic.

    Intentionally approximate — just decides when to trigger compression.
    """
    return sum(len(str(m.content)) for m in messages) // 4


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

    prompt = (
        "You are compressing a financial analysis conversation for context management. "
        "Preserve all specific numbers, tickers, dates, analysis conclusions, open "
        "questions, and anything the user might want to reference later. Be concise "
        "but complete — this summary replaces the full history in future LLM calls.\n\n"
        + "\n\n".join(history_lines)
        + "\n\nSUMMARY:"
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
