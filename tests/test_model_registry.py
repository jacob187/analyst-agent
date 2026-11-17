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
from agents.llm_factory import _PROVIDER_KEY_KWARG
from api.dependencies import _PROVIDER_KEY_FIELDS, PROVIDER_ENV_VARS
from tests.conftest import flagged_default_model, flagged_free_tier_model, other_model

# Product decisions (which model is the default, the free tier, lendable…)
# live in models.json; tests read them from there rather than restating them.
_RAW = json.loads((Path(__file__).parent.parent / "agents" / "models.json").read_text())
_PROVIDERS = sorted({entry["provider"] for entry in _RAW})


@pytest.fixture
def no_env_overrides(monkeypatch):
    """Drop the env overrides so lookups reflect the models.json flags."""
    monkeypatch.delenv("DEFAULT_MODEL_ID", raising=False)
    monkeypatch.delenv("ANON_FREE_MODEL_ID", raising=False)


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

    @pytest.mark.parametrize("provider", _PROVIDERS)
    def test_every_provider_is_wired_for_keys(self, provider):
        # A provider with no env var, key field or factory key kwarg would
        # silently resolve no key (or pass it under the wrong name).
        assert provider in PROVIDER_ENV_VARS
        assert provider in _PROVIDER_KEY_FIELDS
        assert provider in _PROVIDER_KEY_KWARG

    def test_all_ids_are_unique(self):
        # From the raw JSON: the registry dict would silently keep one of a duplicate pair.
        ids = [e["id"] for e in _RAW]
        assert len(ids) == len(set(ids)), "Duplicate model IDs found"

    def test_at_most_one_free_tier_model(self):
        free_tier_models = [m for m in get_all_models() if m.free_tier]
        assert len(free_tier_models) <= 1, f"Expected ≤1 free_tier model, got {len(free_tier_models)}"


class TestGetFreeTierModel:
    def test_returns_the_flagged_model(self, no_env_overrides):
        flagged = [e["id"] for e in _RAW if e.get("free_tier")]
        model = get_free_tier_model()
        assert ([model.id] if model else []) == flagged

    def test_env_override(self, monkeypatch):
        override = other_model(than=flagged_free_tier_model())
        monkeypatch.setenv("ANON_FREE_MODEL_ID", override.id)
        assert get_free_tier_model() == override

    def test_invalid_env_override_falls_back(self, monkeypatch):
        monkeypatch.setenv("ANON_FREE_MODEL_ID", "unknown-model")
        assert get_free_tier_model() == flagged_free_tier_model()

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
    @pytest.mark.parametrize("entry", _RAW, ids=lambda e: e["id"])
    def test_every_json_entry_is_found_by_id(self, entry):
        model = get_model(entry["id"])
        assert (model.id, model.provider) == (entry["id"], entry["provider"])

    def test_unknown_model_returns_none(self):
        assert get_model("unknown-model") is None


class TestGetDefaultModel:
    def test_returns_the_flagged_model(self, no_env_overrides):
        [flagged] = [e["id"] for e in _RAW if e["default"]]
        assert get_default_model().id == flagged

    def test_env_override(self, monkeypatch):
        override = other_model(than=flagged_default_model())
        monkeypatch.setenv("DEFAULT_MODEL_ID", override.id)
        assert get_default_model() == override

    def test_invalid_env_override_falls_back(self, monkeypatch):
        monkeypatch.setenv("DEFAULT_MODEL_ID", "unknown-model")
        assert get_default_model() == flagged_default_model()


class TestGetModelsByProvider:
    @pytest.mark.parametrize("provider", _PROVIDERS)
    def test_returns_exactly_that_providers_models(self, provider):
        expected = [e["id"] for e in _RAW if e["provider"] == provider]
        assert [m.id for m in get_models_by_provider(provider)] == expected

    def test_unknown_provider_returns_empty(self):
        assert get_models_by_provider("unknown_provider") == []


class TestGetTokenThreshold:
    @pytest.mark.parametrize("entry", _RAW, ids=lambda e: e["id"])
    def test_threshold_is_a_quarter_of_the_context_window(self, entry):
        assert get_token_threshold(entry["id"]) == entry["max_context"] // 4

    def test_unknown_model_returns_fallback(self):
        assert get_token_threshold("unknown-model") == 30_000


class TestModelDefImmutability:
    def test_model_is_frozen(self):
        model = get_default_model()
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
        for entry in _RAW:
            for key in ("operator_lendable", *_PRICE_KEYS):
                assert key in entry, f"{entry['id']} is missing {key}"

    def test_free_tier_models_are_lendable(self):
        # Anonymous visitors run the free-tier model on the operator's key, so
        # signed-in users must be able to as well.
        assert all(e["operator_lendable"] for e in _RAW if e.get("free_tier"))

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
