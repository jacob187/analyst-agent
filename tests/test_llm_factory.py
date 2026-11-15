"""Tests for agents/llm_factory.py — LLM creation with provider abstraction.

Models come from the registry (agents/models.json) and the factory's own
Anthropic quirk sets, so adding, removing or re-defaulting a model doesn't
need a test edit. Only `init_chat_model` (the provider boundary) is mocked.
"""

from unittest.mock import MagicMock, patch

import pytest

from agents.llm_factory import (
    _ANTHROPIC_LEGACY_BUDGET_MODELS,
    _ANTHROPIC_SAMPLING_MODELS,
    _PROVIDER_KEY_KWARG,
    ThinkingConfig,
    _build_thinking_kwargs,
    create_llm,
    create_llm_pair,
)
from agents.model_registry import get_all_models, get_default_model, get_models_by_provider
from tests.conftest import model_where

_ALL_MODELS = get_all_models()
_ANTHROPIC_IDS = {m.id for m in get_models_by_provider("anthropic")}
_ADAPTIVE_THINKING_IDS = sorted(_ANTHROPIC_IDS - _ANTHROPIC_LEGACY_BUDGET_MODELS)
_NO_SAMPLING_IDS = sorted(_ANTHROPIC_IDS - _ANTHROPIC_SAMPLING_MODELS)

# The kwargs each provider's thinking/reasoning setting travels under.
_THINKING_KWARGS = {
    "google_genai": {"thinking_level", "include_thoughts"},
    "anthropic": {"thinking"},
    "openai": {"reasoning_effort"},
}
_ANY_THINKING_KWARG = set().union(*_THINKING_KWARGS.values())
_ON = ThinkingConfig(enabled=True, level="medium")


def _by_id(model):
    return model.id


class TestThinkingConfig:
    def test_defaults(self):
        config = ThinkingConfig()
        assert config.enabled is False
        assert config.level == "medium"

    def test_frozen(self):
        config = ThinkingConfig(enabled=True, level="high")
        with pytest.raises(Exception):
            config.enabled = False


class TestAnthropicQuirkSets:
    @pytest.mark.parametrize(
        "quirk_set", [_ANTHROPIC_LEGACY_BUDGET_MODELS, _ANTHROPIC_SAMPLING_MODELS],
        ids=["legacy_budget", "sampling"],
    )
    def test_names_only_registry_anthropic_models(self, quirk_set):
        # A stale id here (a model renamed or dropped from models.json) means
        # the factory no longer special-cases the model it was meant for.
        assert quirk_set <= _ANTHROPIC_IDS


class TestBuildThinkingKwargs:
    def test_none_config_returns_empty(self):
        assert _build_thinking_kwargs("google_genai", "any-model", None) == {}

    def test_disabled_config_returns_empty(self):
        config = ThinkingConfig(enabled=False, level="high")
        assert _build_thinking_kwargs("openai", "any-model", config) == {}

    @pytest.mark.parametrize("level", ["low", "medium"])
    def test_google_genai(self, level):
        config = ThinkingConfig(enabled=True, level=level)
        result = _build_thinking_kwargs("google_genai", "any-model", config)
        assert result == {"thinking_level": level, "include_thoughts": True}

    @pytest.mark.parametrize("level,budget", [("medium", 4096), ("high", 8192)])
    @pytest.mark.parametrize("model_id", sorted(_ANTHROPIC_LEGACY_BUDGET_MODELS))
    def test_anthropic_legacy_model_uses_budget_tokens(self, model_id, level, budget):
        config = ThinkingConfig(enabled=True, level=level)
        result = _build_thinking_kwargs("anthropic", model_id, config)
        assert result == {"thinking": {"type": "enabled", "budget_tokens": budget}}

    @pytest.mark.parametrize("model_id", _ADAPTIVE_THINKING_IDS)
    def test_anthropic_non_legacy_model_uses_adaptive(self, model_id):
        """Opus 4.7+/Claude 5 models reject the legacy budget_tokens format —
        anything not on the legacy allowlist gets adaptive thinking instead."""
        config = ThinkingConfig(enabled=True, level="high")
        result = _build_thinking_kwargs("anthropic", model_id, config)
        assert result == {"thinking": {"type": "adaptive"}}

    def test_openai(self):
        result = _build_thinking_kwargs("openai", "any-model", _ON)
        assert result == {"reasoning_effort": "medium"}

    def test_unknown_provider_returns_empty(self):
        assert _build_thinking_kwargs("unknown", "any-model", _ON) == {}


@patch("langchain.chat_models.init_chat_model", return_value=MagicMock())
class TestCreateLlmContract:
    """Every registry model builds, whatever its provider."""

    @pytest.mark.parametrize("model", _ALL_MODELS, ids=_by_id)
    def test_passes_provider_prefixed_id_and_key(self, mock_init, model):
        create_llm(model.id, "fake-key")

        args, kwargs = mock_init.call_args
        assert args == (f"{model.provider}:{model.id}",)
        assert kwargs[_PROVIDER_KEY_KWARG[model.provider]] == "fake-key"

    @pytest.mark.parametrize("model", _ALL_MODELS, ids=_by_id)
    def test_sets_max_tokens_only_for_anthropic(self, mock_init, model):
        # langchain-anthropic falls back to 4096 output tokens for models it
        # has no profile for — too few once adaptive thinking spends some.
        create_llm(model.id, "fake-key")

        expected = 16_000 if model.provider == "anthropic" else None
        assert mock_init.call_args.kwargs.get("max_tokens") == expected

    @pytest.mark.parametrize("model", _ALL_MODELS, ids=_by_id)
    def test_thinking_kwargs_follow_provider_and_capability(self, mock_init, model):
        create_llm(model.id, "fake-key", thinking=_ON)

        sent = mock_init.call_args.kwargs.keys() & _ANY_THINKING_KWARG
        assert sent == (_THINKING_KWARGS[model.provider] if model.thinking_capable else set())

    @pytest.mark.parametrize("model", _ALL_MODELS, ids=_by_id)
    def test_no_thinking_kwargs_without_thinking(self, mock_init, model):
        create_llm(model.id, "fake-key")

        assert not mock_init.call_args.kwargs.keys() & _ANY_THINKING_KWARG

    @pytest.mark.parametrize(
        "model", [m for m in _ALL_MODELS if m.provider != "anthropic"], ids=_by_id
    )
    def test_non_anthropic_models_get_zero_temperature(self, mock_init, model):
        """gpt-5 (non-chat) rejects a non-default temperature, but dropping it
        is langchain-openai's job — ChatOpenAI.validate_temperature pops it, and
        the responses-API payload builder pops it again."""
        create_llm(model.id, "fake-key", thinking=_ON)

        assert mock_init.call_args.kwargs["temperature"] == 0


@patch("langchain.chat_models.init_chat_model", return_value=MagicMock())
class TestCreateLlmAnthropicSampling:
    @pytest.mark.parametrize("model_id", sorted(_ANTHROPIC_SAMPLING_MODELS))
    def test_sampling_model_passes_temperature(self, mock_init, model_id):
        create_llm(model_id, "fake-anthropic-key")

        mock_init.assert_called_once_with(
            f"anthropic:{model_id}",
            temperature=0,
            anthropic_api_key="fake-anthropic-key",
            max_tokens=16_000,
        )

    @pytest.mark.parametrize("model_id", _NO_SAMPLING_IDS)
    def test_model_without_sampling_omits_temperature(self, mock_init, model_id):
        # Opus 4.7+ and Claude 5 reject any temperature with a 400.
        create_llm(model_id, "fake-anthropic-key")

        mock_init.assert_called_once_with(
            f"anthropic:{model_id}",
            anthropic_api_key="fake-anthropic-key",
            max_tokens=16_000,
        )

    @pytest.mark.parametrize("model_id", sorted(_ANTHROPIC_SAMPLING_MODELS))
    def test_thinking_omits_temperature(self, mock_init, model_id):
        # Sampling models take temperature, but not alongside thinking.
        create_llm(model_id, "fake-anthropic-key", thinking=_ON)

        kwargs = mock_init.call_args.kwargs
        assert "thinking" in kwargs
        assert "temperature" not in kwargs


class TestCreateLlm:
    @patch("langchain.chat_models.init_chat_model")
    def test_unknown_model_falls_back_to_default(self, mock_init):
        mock_init.return_value = MagicMock()
        create_llm("unknown-model", "fake-key")

        default = get_default_model()
        assert mock_init.call_args.args == (f"{default.provider}:{default.id}",)


class TestCreateLlmPair:
    @patch("langchain.chat_models.init_chat_model")
    def test_thinking_capable_model_returns_two_distinct_llms(self, mock_init):
        """For a thinking-capable model, the pair should be two separate instances."""
        model = model_where(thinking_capable=True)
        mock_init.side_effect = [MagicMock(name="main"), MagicMock(name="synth")]
        main, synth = create_llm_pair(model.id, "fake-key")

        assert main is not synth
        assert mock_init.call_count == 2

        # Main LLM without thinking, synthesizer with it.
        first_call_kwargs, second_call_kwargs = (c.kwargs for c in mock_init.call_args_list)
        assert not first_call_kwargs.keys() & _ANY_THINKING_KWARG
        assert second_call_kwargs.keys() & _ANY_THINKING_KWARG

    @patch("langchain.chat_models.init_chat_model")
    def test_non_thinking_model_returns_same_instance(self, mock_init):
        """For a non-thinking model, both LLMs should be the same instance."""
        model = model_where(thinking_capable=False)
        mock_llm = MagicMock()
        mock_init.return_value = mock_llm
        main, synth = create_llm_pair(model.id, "fake-key")

        assert main is synth
        assert mock_init.call_count == 1
