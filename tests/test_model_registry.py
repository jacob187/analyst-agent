"""Tests for agents/model_registry.py — model loading, validation, and lookups."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from agents.model_registry import (
    ModelDef,
    _build_registry,
    get_all_models,
    get_default_model,
    get_free_tier_model,
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

    def test_at_most_one_free_tier_model(self):
        free_tier_models = [m for m in get_all_models() if m.free_tier]
        assert len(free_tier_models) <= 1, f"Expected ≤1 free_tier model, got {len(free_tier_models)}"


class TestGetFreeTierModel:
    def test_returns_the_flagged_model(self):
        model = get_free_tier_model()
        assert model.id == "gemini-3.1-flash-lite"
        assert model.free_tier is True

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("ANON_FREE_MODEL_ID", "gpt-4.1-mini")
        assert get_free_tier_model().id == "gpt-4.1-mini"

    def test_invalid_env_override_falls_back(self, monkeypatch):
        monkeypatch.setenv("ANON_FREE_MODEL_ID", "nonexistent-model-xyz")
        assert get_free_tier_model().id == "gemini-3.1-flash-lite"

    def test_returns_none_when_no_model_flagged(self, monkeypatch):
        # Fails closed: unflagging the free-tier model disables the trial rather
        # than re-pointing the operator's key at whatever the default model is.
        import agents.model_registry as registry
        monkeypatch.delenv("ANON_FREE_MODEL_ID", raising=False)
        monkeypatch.setattr(registry, "_FREE_TIER_MODEL", None)
        assert get_free_tier_model() is None

    def test_max_context_is_positive(self):
        for model in get_all_models():
            assert model.max_context > 0, f"max_context must be >0 for {model.id}"


class TestGetModel:
    def test_known_model(self):
        model = get_model("gemini-3.6-flash")
        assert model is not None
        assert model.id == "gemini-3.6-flash"
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
        assert get_default_model().id == "gemini-3.6-flash"


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
        assert get_token_threshold("gemini-3.6-flash") == 250_000

    def test_gpt41_mini_threshold(self):
        # 1_000_000 // 4 = 250_000
        assert get_token_threshold("gpt-4.1-mini") == 250_000

    def test_gpt54_mini_threshold(self):
        # 400_000 // 4 = 100_000
        assert get_token_threshold("gpt-5.4-mini") == 100_000

    def test_unknown_model_returns_fallback(self):
        assert get_token_threshold("nonexistent-model") == 30_000

    def test_threshold_is_positive(self):
        for model in get_all_models():
            threshold = get_token_threshold(model.id)
            assert threshold > 0


class TestModelDefImmutability:
    def test_model_is_frozen(self):
        model = get_model("gemini-3.6-flash")
        with pytest.raises(Exception):
            model.id = "something-else"


# ---------------------------------------------------------------------------
# operator_lendable + pricing fields
# ---------------------------------------------------------------------------

_PRICE_KEYS = ("input_price_per_mtok", "output_price_per_mtok", "prices_as_of")


def _entry(**overrides):
    base = {
        "id": "m1", "provider": "google_genai", "display_name": "M1",
        "max_context": 1000, "thinking_capable": False, "default": True,
    }
    return {**base, **overrides}


class TestOperatorLendable:
    def test_every_json_entry_declares_lendable_and_price_fields(self):
        raw = json.loads((Path(__file__).parent.parent / "agents" / "models.json").read_text())
        for entry in raw:
            for key in ("operator_lendable", *_PRICE_KEYS):
                assert key in entry, f"{entry['id']} is missing {key}"

    def test_only_default_and_free_tier_are_lendable(self):
        lendable = {m.id for m in get_all_models() if m.operator_lendable}
        assert lendable == {"gemini-3.6-flash", "gemini-3.1-flash-lite"}

    def test_lendable_defaults_to_false(self):
        models, _, _ = _build_registry([_entry()])
        assert models["m1"].operator_lendable is False

    def test_free_tier_model_must_be_lendable(self):
        with pytest.raises(ValueError, match="operator_lendable"):
            _build_registry([_entry(), _entry(id="m2", default=False, free_tier=True)])

    def test_free_tier_lendable_model_loads(self):
        _, _, free = _build_registry(
            [_entry(), _entry(id="m2", default=False, free_tier=True, operator_lendable=True)]
        )
        assert free.id == "m2"

    @pytest.mark.parametrize("bad", ["true", 1, None])
    def test_lendable_must_be_a_real_bool(self, bad):
        with pytest.raises(ValidationError):
            _build_registry([_entry(operator_lendable=bad)])


class TestPriceFields:
    def test_prices_are_all_set_or_all_null_per_model(self):
        # A model either has a complete price (input, output, as-of date) or
        # none at all — a half-filled price would yield a wrong cost estimate.
        for model in get_all_models():
            values = [getattr(model, key) for key in _PRICE_KEYS]
            assert all(v is None for v in values) or all(v is not None for v in values), model.id

    def test_valid_prices_load(self):
        models, _, _ = _build_registry([_entry(
            input_price_per_mtok=3, output_price_per_mtok=15.5, prices_as_of="2026-11-01",
        )])
        m = models["m1"]
        assert (m.input_price_per_mtok, m.output_price_per_mtok, m.prices_as_of) == (3.0, 15.5, "2026-11-01")

    @pytest.mark.parametrize("bad", ["3", True, -1])
    def test_bad_price_rejected(self, bad):
        with pytest.raises(ValidationError):
            _build_registry([_entry(input_price_per_mtok=bad)])

    @pytest.mark.parametrize("bad", ["2026-13-01", "Nov 1 2026", "20261101", "2026-W44-1", 20261101])
    def test_bad_prices_as_of_rejected(self, bad):
        with pytest.raises(ValidationError):
            _build_registry([_entry(prices_as_of=bad)])
