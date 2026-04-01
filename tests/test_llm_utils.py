"""Tests for agents/llm_utils.py — reusable thinking extraction module."""

from unittest.mock import MagicMock

from agents.llm_utils import LLMResponse, extract_text, extract_thinking, parse_llm_response


class TestParseStringContent:
    def test_plain_string(self):
        result = parse_llm_response("Hello world")
        assert result == LLMResponse(text="Hello world", thinking="")

    def test_aimessage_string_content(self):
        msg = MagicMock()
        msg.content = "Analysis complete."
        result = parse_llm_response(msg)
        assert result.text == "Analysis complete."
        assert result.thinking == ""


class TestParseListContent:
    def test_text_only_blocks(self):
        msg = MagicMock()
        msg.content = [
            {"type": "text", "text": "First paragraph."},
            {"type": "text", "text": " Second paragraph."},
        ]
        result = parse_llm_response(msg)
        assert result.text == "First paragraph. Second paragraph."
        assert result.thinking == ""

    def test_thinking_and_text_blocks(self):
        msg = MagicMock()
        msg.content = [
            {"type": "thinking", "thinking": "Let me analyze the RSI values..."},
            {"type": "text", "text": '{"market_regime": "bull"}'},
        ]
        result = parse_llm_response(msg)
        assert result.text == '{"market_regime": "bull"}'
        assert result.thinking == "Let me analyze the RSI values..."

    def test_reasoning_block_variant(self):
        """Gemini v1 format uses 'reasoning' instead of 'thinking'."""
        msg = MagicMock()
        msg.content = [
            {"type": "reasoning", "reasoning": "Step by step analysis..."},
            {"type": "text", "text": "Final answer."},
        ]
        result = parse_llm_response(msg)
        assert result.text == "Final answer."
        assert result.thinking == "Step by step analysis..."

    def test_multiple_thinking_blocks(self):
        msg = MagicMock()
        msg.content = [
            {"type": "thinking", "thinking": "First thought."},
            {"type": "thinking", "thinking": "Second thought."},
            {"type": "text", "text": "Output."},
        ]
        result = parse_llm_response(msg)
        assert result.thinking == "First thought.\n\nSecond thought."
        assert result.text == "Output."

    def test_empty_thinking_block_ignored(self):
        msg = MagicMock()
        msg.content = [
            {"type": "thinking", "thinking": ""},
            {"type": "text", "text": "Output."},
        ]
        result = parse_llm_response(msg)
        assert result.thinking == ""

    def test_string_elements_in_list(self):
        msg = MagicMock()
        msg.content = ["raw string", {"type": "text", "text": " more text"}]
        result = parse_llm_response(msg)
        assert result.text == "raw string more text"

    def test_unknown_block_type_ignored(self):
        msg = MagicMock()
        msg.content = [
            {"type": "metadata", "data": "something"},
            {"type": "text", "text": "Output."},
        ]
        result = parse_llm_response(msg)
        assert result.text == "Output."
        assert result.thinking == ""


class TestParseFallback:
    def test_integer_content(self):
        result = parse_llm_response(42)
        assert result.text == "42"
        assert result.thinking == ""

    def test_none_content(self):
        result = parse_llm_response(None)
        assert result.text == "None"
        assert result.thinking == ""


class TestConvenienceFunctions:
    def test_extract_text(self):
        msg = MagicMock()
        msg.content = [
            {"type": "thinking", "thinking": "reasoning..."},
            {"type": "text", "text": "answer"},
        ]
        assert extract_text(msg) == "answer"

    def test_extract_thinking(self):
        msg = MagicMock()
        msg.content = [
            {"type": "thinking", "thinking": "reasoning..."},
            {"type": "text", "text": "answer"},
        ]
        assert extract_thinking(msg) == "reasoning..."

    def test_extract_thinking_from_plain_string(self):
        assert extract_thinking("just text") == ""


class TestLLMResponseImmutable:
    def test_frozen_dataclass(self):
        result = LLMResponse(text="a", thinking="b")
        try:
            result.text = "c"
            assert False, "Should have raised FrozenInstanceError"
        except AttributeError:
            pass
