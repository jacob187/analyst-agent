"""Tests for memory management helpers in api/main.py.

Marker: eval_unit (no API keys, no network).

Functions under test:
  - estimate_tokens(messages)          — pure function, deterministic
  - build_context_from_session(...)    — pure function, deterministic
  - compress_history(...)              — async, LLM call mocked

Why mock the LLM in compress_history?
  The function's job is to call the LLM, stitch the result into the right
  structure, and persist the summary. We test the *orchestration logic*, not
  the quality of the LLM output — that belongs in an eval test with a real key.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage

from api.memory import (
    estimate_tokens,
    build_context_from_session,
    compress_history,
    RECENT_MESSAGES_TO_KEEP,
    TOKEN_THRESHOLD,
)


# ---------------------------------------------------------------------------
# estimate_tokens
# ---------------------------------------------------------------------------

@pytest.mark.eval_unit
class TestEstimateTokens:
    """
    estimate_tokens uses chars/4 as a rough heuristic.
    We test the arithmetic directly rather than asserting exact token counts
    from a real tokeniser — the point is that the function is monotonically
    increasing and returns 0 for an empty list.
    """

    def test_empty_list_returns_zero(self):
        assert estimate_tokens([]) == 0

    def test_single_short_message(self):
        # "hello" = 5 chars → 5 // 4 = 1
        msgs = [HumanMessage(content="hello")]
        assert estimate_tokens(msgs) == 1

    def test_scales_with_content_length(self):
        short = [HumanMessage(content="A" * 100)]
        long = [HumanMessage(content="A" * 1000)]
        assert estimate_tokens(long) > estimate_tokens(short)

    def test_sum_across_multiple_messages(self):
        # 400 chars each, 2 messages → 800 chars → 200 tokens
        msgs = [
            HumanMessage(content="A" * 400),
            AIMessage(content="B" * 400),
        ]
        assert estimate_tokens(msgs) == 200

    def test_token_threshold_is_reachable(self):
        """A long conversation must be able to exceed TOKEN_THRESHOLD."""
        # Each message is 4000 chars ≈ 1000 tokens; 50 messages ≈ 50K tokens
        msgs = [HumanMessage(content="X" * 4000) for _ in range(50)]
        assert estimate_tokens(msgs) > TOKEN_THRESHOLD


# ---------------------------------------------------------------------------
# build_context_from_session
# ---------------------------------------------------------------------------

def _session(summary: str | None) -> dict:
    return {"id": "s1", "ticker": "AAPL", "summary": summary, "created_at": "2026-01-01"}


def _msgs(n: int) -> list[dict]:
    """Alternating user/assistant messages."""
    roles = ["user", "assistant"]
    return [{"role": roles[i % 2], "content": f"msg {i}"} for i in range(n)]


@pytest.mark.eval_unit
class TestBuildContextFromSession:
    def test_no_summary_loads_all_messages(self):
        history = build_context_from_session(_session(None), _msgs(4))
        assert len(history) == 4

    def test_message_types_map_correctly(self):
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ]
        history = build_context_from_session(_session(None), msgs)
        assert isinstance(history[0], HumanMessage)
        assert isinstance(history[1], AIMessage)

    def test_with_summary_and_many_messages_uses_summary_plus_recent(self):
        """
        When a summary exists and message count > RECENT_MESSAGES_TO_KEEP,
        the result should be: [summary_message] + last RECENT_MESSAGES_TO_KEEP messages.
        """
        n = RECENT_MESSAGES_TO_KEEP + 4  # clearly more than the threshold
        history = build_context_from_session(_session("Prior analysis."), _msgs(n))

        assert len(history) == RECENT_MESSAGES_TO_KEEP + 1
        assert "[CONVERSATION SUMMARY]" in history[0].content
        assert "Prior analysis." in history[0].content

    def test_recent_messages_are_the_last_n(self):
        """The verbatim tail must be the *last* RECENT_MESSAGES_TO_KEEP messages."""
        all_msgs = _msgs(RECENT_MESSAGES_TO_KEEP + 2)
        history = build_context_from_session(_session("summary"), all_msgs)
        expected_recent_contents = [m["content"] for m in all_msgs[-RECENT_MESSAGES_TO_KEEP:]]
        actual_recent_contents = [m.content for m in history[1:]]
        assert actual_recent_contents == expected_recent_contents

    def test_with_summary_but_few_messages_loads_all(self):
        """
        If message count <= RECENT_MESSAGES_TO_KEEP, we load verbatim even
        when a summary exists — not worth the extra context overhead.
        """
        msgs = _msgs(2)  # well below threshold
        history = build_context_from_session(_session("some summary"), msgs)
        assert len(history) == 2
        # First message must NOT be a summary placeholder
        assert not history[0].content.startswith("[CONVERSATION SUMMARY]")

    def test_empty_messages_returns_empty_list(self):
        history = build_context_from_session(_session(None), [])
        assert history == []

    def test_none_session_loads_all_messages(self):
        """Passing session=None (e.g. DB lookup failed) is safe."""
        history = build_context_from_session(None, _msgs(3))
        assert len(history) == 3


# ---------------------------------------------------------------------------
# compress_history
# ---------------------------------------------------------------------------

def _make_history(n: int) -> list:
    """Alternating HumanMessage/AIMessage, length n."""
    return [
        HumanMessage(content=f"user turn {i}") if i % 2 == 0
        else AIMessage(content=f"assistant turn {i}")
        for i in range(n)
    ]


@pytest.fixture
def mock_llm():
    """LLM whose ainvoke returns a fixed summary string."""
    llm = AsyncMock()
    response = MagicMock()
    response.content = "Summarised: user asked about AAPL financials."
    llm.ainvoke.return_value = response
    return llm


@pytest.mark.eval_unit
class TestCompressHistory:
    async def test_returns_summary_plus_recent(self, mock_llm):
        history = _make_history(RECENT_MESSAGES_TO_KEEP + 4)
        with patch("api.memory.update_session_summary"):
            result = await compress_history("session-1", history, mock_llm)

        assert len(result) == RECENT_MESSAGES_TO_KEEP + 1

    async def test_first_message_is_summary(self, mock_llm):
        history = _make_history(RECENT_MESSAGES_TO_KEEP + 2)
        with patch("api.memory.update_session_summary"):
            result = await compress_history("session-1", history, mock_llm)

        assert result[0].content.startswith("[CONVERSATION SUMMARY]")
        assert "Summarised" in result[0].content

    async def test_recent_messages_are_preserved_verbatim(self, mock_llm):
        history = _make_history(RECENT_MESSAGES_TO_KEEP + 4)
        with patch("api.memory.update_session_summary"):
            result = await compress_history("session-1", history, mock_llm)

        # Tail of result (after the summary message) must match tail of input
        assert result[1:] == history[-RECENT_MESSAGES_TO_KEEP:]

    async def test_llm_receives_transcript_of_old_messages(self, mock_llm):
        """
        The messages that are NOT in the recent tail must appear in the
        prompt sent to the LLM.
        """
        history = _make_history(RECENT_MESSAGES_TO_KEEP + 2)
        old_messages = history[:-RECENT_MESSAGES_TO_KEEP]
        with patch("api.memory.update_session_summary"):
            await compress_history("session-1", history, mock_llm)

        call_args = mock_llm.ainvoke.call_args[0][0]  # list[HumanMessage]
        prompt_text = call_args[0].content
        # Each old message's content must appear in the summarisation prompt
        for msg in old_messages:
            assert str(msg.content) in prompt_text

    async def test_nested_summary_is_collapsed(self, mock_llm):
        """
        If the history already starts with a [CONVERSATION SUMMARY] message,
        it must be reformatted as 'PRIOR SUMMARY:' so we don't nest summaries.
        """
        prior_summary = HumanMessage(content="[CONVERSATION SUMMARY]\nEarlier: RSI analysis.")
        recent = _make_history(RECENT_MESSAGES_TO_KEEP)
        history = [prior_summary] + recent

        with patch("api.memory.update_session_summary"):
            await compress_history("session-1", history, mock_llm)

        prompt_text = mock_llm.ainvoke.call_args[0][0][0].content
        # Should NOT appear verbatim — must be reformatted
        assert "[CONVERSATION SUMMARY]" not in prompt_text
        assert "PRIOR SUMMARY:" in prompt_text

    async def test_persists_summary_to_db(self, mock_llm):
        """update_session_summary must be scheduled (via create_task)."""
        history = _make_history(RECENT_MESSAGES_TO_KEEP + 2)
        with patch("api.memory.update_session_summary") as mock_save, \
             patch("api.memory.asyncio.create_task") as mock_task:
            await compress_history("session-42", history, mock_llm)

        # create_task should have been called once (with the coroutine from update_session_summary)
        mock_task.assert_called_once()

    async def test_result_is_human_message_type(self, mock_llm):
        """The summary placeholder must be a HumanMessage so the LLM reads it as context."""
        history = _make_history(RECENT_MESSAGES_TO_KEEP + 2)
        with patch("api.memory.update_session_summary"), \
             patch("api.memory.asyncio.create_task"):
            result = await compress_history("session-1", history, mock_llm)

        assert isinstance(result[0], HumanMessage)
