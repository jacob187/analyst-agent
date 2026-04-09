"""Tests for agents/model_registry.py — model loading, validation, and lookups."""

import pytest

from agents.model_registry import (
    ModelDef,
    get_all_models,
    get_default_model,
    get_model,
    get_models_by_provider,
    get_token_threshold,
)


class TestModelsJsonLoading:
    """Verify that models.json loads correctly and passes validation."""

    def test_loads_at_least_three_models(self):
        models = get_all_models()
        assert len(models) >= 3, f"Expected ≥3 models, got {len(models)}"

    def test_all_entries_are_model_def(self):
        for model in get_all_models():
            assert isinstance(model, ModelDef)

    def test_exactly_one_default(self):
        defaults = [m for m in get_all_models() if m.default]
        assert len(defaults) == 1, f"Expected 1 default, got {len(defaults)}"

    def test_all_providers_are_known(self):
        allowed = {"google_genai", "openai", "anthropic"}
        for model in get_all_models():
            assert model.provider in allowed, (
                f"Unknown provider '{model.provider}' on model '{model.id}'"
            )

    def test_all_ids_are_unique(self):
        ids = [m.id for m in get_all_models()]
        assert len(ids) == len(set(ids)), "Duplicate model IDs found"

    def test_max_context_is_positive(self):
        for model in get_all_models():
            assert model.max_context > 0, f"max_context must be >0 for {model.id}"


class TestGetModel:
    def test_known_model(self):
        model = get_model("gemini-3-flash-preview")
        assert model is not None
        assert model.id == "gemini-3-flash-preview"
        assert model.provider == "google_genai"

    def test_unknown_model_returns_none(self):
        assert get_model("nonexistent-model-xyz") is None


class TestGetDefaultModel:
    def test_returns_valid_model(self):
        default = get_default_model()
        assert isinstance(default, ModelDef)
        assert default.default is True

    def test_default_is_gemini_flash(self):
        """Current default — update this test if the default changes."""
        assert get_default_model().id == "gemini-3-flash-preview"


class TestGetModelsByProvider:
    def test_google_models(self):
        models = get_models_by_provider("google_genai")
        assert len(models) >= 1
        assert all(m.provider == "google_genai" for m in models)

    def test_openai_models(self):
        models = get_models_by_provider("openai")
        assert len(models) >= 1
        assert all(m.provider == "openai" for m in models)

    def test_unknown_provider_returns_empty(self):
        assert get_models_by_provider("unknown_provider") == []


class TestGetTokenThreshold:
    def test_gemini_flash_threshold(self):
        # 1_000_000 // 4 = 250_000
        assert get_token_threshold("gemini-3-flash-preview") == 250_000

    def test_gpt41_mini_threshold(self):
        # 1_000_000 // 4 = 250_000
        assert get_token_threshold("gpt-4.1-mini") == 250_000

    def test_unknown_model_returns_fallback(self):
        assert get_token_threshold("nonexistent-model") == 30_000

    def test_threshold_is_positive(self):
        for model in get_all_models():
            threshold = get_token_threshold(model.id)
            assert threshold > 0


class TestModelDefImmutability:
    def test_model_is_frozen(self):
        model = get_model("gemini-3-flash-preview")
        with pytest.raises(Exception):
            model.id = "something-else"
