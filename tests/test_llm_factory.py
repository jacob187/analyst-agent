"""Tests for agents/llm_factory.py — LLM creation with provider abstraction."""

from unittest.mock import MagicMock, patch

import pytest

from agents.llm_factory import (
    ThinkingConfig,
    _build_thinking_kwargs,
    create_llm,
    create_llm_pair,
)


class TestThinkingConfig:
    def test_defaults(self):
        config = ThinkingConfig()
        assert config.enabled is False
        assert config.level == "medium"

    def test_frozen(self):
        config = ThinkingConfig(enabled=True, level="high")
        with pytest.raises(Exception):
            config.enabled = False


class TestBuildThinkingKwargs:
    def test_none_config_returns_empty(self):
        assert _build_thinking_kwargs("google_genai", "gemini-3-flash-preview", None) == {}

    def test_disabled_config_returns_empty(self):
        config = ThinkingConfig(enabled=False, level="high")
        assert _build_thinking_kwargs("openai", "gpt-4.1-mini", config) == {}

    def test_google_genai(self):
        config = ThinkingConfig(enabled=True, level="medium")
        result = _build_thinking_kwargs("google_genai", "gemini-3-flash-preview", config)
        assert result == {"thinking_level": "medium", "include_thoughts": True}

    def test_google_genai_low(self):
        config = ThinkingConfig(enabled=True, level="low")
        result = _build_thinking_kwargs("google_genai", "gemini-3-flash-preview", config)
        assert result == {"thinking_level": "low", "include_thoughts": True}

    def test_anthropic(self):
        config = ThinkingConfig(enabled=True, level="medium")
        result = _build_thinking_kwargs("anthropic", "claude-sonnet-4-6", config)
        assert result == {"thinking": {"type": "enabled", "budget_tokens": 4096}}

    def test_anthropic_high(self):
        config = ThinkingConfig(enabled=True, level="high")
        result = _build_thinking_kwargs("anthropic", "claude-sonnet-4-6", config)
        assert result == {"thinking": {"type": "enabled", "budget_tokens": 8192}}

    def test_openai(self):
        config = ThinkingConfig(enabled=True, level="medium")
        result = _build_thinking_kwargs("openai", "o3", config)
        assert result == {"reasoning_effort": "medium"}

    def test_unknown_provider_returns_empty(self):
        config = ThinkingConfig(enabled=True, level="medium")
        assert _build_thinking_kwargs("unknown", "some-model", config) == {}


class TestCreateLlm:
    @patch("langchain.chat_models.init_chat_model")
    def test_google_model(self, mock_init):
        mock_init.return_value = MagicMock()
        create_llm("gemini-3-flash-preview", "fake-google-key")

        mock_init.assert_called_once_with(
            "google_genai:gemini-3-flash-preview",
            temperature=0,
            google_api_key="fake-google-key",
        )

    @patch("langchain.chat_models.init_chat_model")
    def test_openai_model(self, mock_init):
        mock_init.return_value = MagicMock()
        create_llm("gpt-4.1-mini", "fake-openai-key")

        mock_init.assert_called_once_with(
            "openai:gpt-4.1-mini",
            temperature=0,
            api_key="fake-openai-key",
        )

    @patch("langchain.chat_models.init_chat_model")
    def test_anthropic_model(self, mock_init):
        mock_init.return_value = MagicMock()
        create_llm("claude-sonnet-4-6", "fake-anthropic-key")

        mock_init.assert_called_once_with(
            "anthropic:claude-sonnet-4-6",
            temperature=0,
            anthropic_api_key="fake-anthropic-key",
        )

    @patch("langchain.chat_models.init_chat_model")
    def test_google_with_thinking(self, mock_init):
        mock_init.return_value = MagicMock()
        thinking = ThinkingConfig(enabled=True, level="medium")
        create_llm("gemini-3-flash-preview", "fake-key", thinking=thinking)

        mock_init.assert_called_once_with(
            "google_genai:gemini-3-flash-preview",
            temperature=0,
            google_api_key="fake-key",
            thinking_level="medium",
            include_thoughts=True,
        )

    @patch("langchain.chat_models.init_chat_model")
    def test_thinking_ignored_for_non_capable_model(self, mock_init):
        """gpt-4.1-mini is not thinking-capable — thinking config should be ignored."""
        mock_init.return_value = MagicMock()
        thinking = ThinkingConfig(enabled=True, level="high")
        create_llm("gpt-4.1-mini", "fake-key", thinking=thinking)

        # Should NOT include reasoning_effort or any thinking kwargs
        mock_init.assert_called_once_with(
            "openai:gpt-4.1-mini",
            temperature=0,
            api_key="fake-key",
        )

    @patch("langchain.chat_models.init_chat_model")
    def test_unknown_model_falls_back_to_default(self, mock_init):
        mock_init.return_value = MagicMock()
        create_llm("nonexistent-model", "fake-key")

        # Should fall back to the default model (gemini-3-flash-preview)
        call_args = mock_init.call_args
        assert "google_genai:gemini-3-flash-preview" in call_args[0]


class TestCreateLlmPair:
    @patch("langchain.chat_models.init_chat_model")
    def test_thinking_capable_model_returns_two_distinct_llms(self, mock_init):
        """For a thinking-capable model, the pair should be two separate instances."""
        mock_init.side_effect = [MagicMock(name="main"), MagicMock(name="synth")]
        main, synth = create_llm_pair("gemini-3-flash-preview", "fake-key")

        assert main is not synth
        assert mock_init.call_count == 2

        # First call: no thinking kwargs
        first_call_kwargs = mock_init.call_args_list[0][1]
        assert "thinking_level" not in first_call_kwargs

        # Second call: thinking kwargs present
        second_call_kwargs = mock_init.call_args_list[1][1]
        assert second_call_kwargs.get("thinking_level") == "medium"

    @patch("langchain.chat_models.init_chat_model")
    def test_non_thinking_model_returns_same_instance(self, mock_init):
        """For a non-thinking model, both LLMs should be the same instance."""
        mock_llm = MagicMock()
        mock_init.return_value = mock_llm
        main, synth = create_llm_pair("gpt-4.1-mini", "fake-key")

        assert main is synth
        assert mock_init.call_count == 1
