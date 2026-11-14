"""`api.dependencies.resolve_model`: the model a request runs on and the key
that may run it, shared by chat and the filings endpoints.

Pure logic over real `ApiKeys` and the real registry — no mocks.
"""

import pytest

from agents.model_registry import get_all_models, get_default_model, get_free_tier_model, get_model
from api import dependencies
from api.dependencies import ApiKeys, resolve_model
from tests.conftest import clear_operator_keys
from tests.test_dependencies import clerk_env, rsa_keypair  # noqa: F401  (fixtures)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    clear_operator_keys(monkeypatch)
    monkeypatch.delenv("DEFAULT_MODEL_ID", raising=False)


def _keys(*, byok: dict[str, str] | None = None, env: dict[str, str] | None = None,
          model_id: str | None = None) -> ApiKeys:
    """ApiKeys with per-provider keys marked as caller-supplied or operator env."""
    fields = {"google_genai": "google_api_key", "openai": "openai_api_key", "anthropic": "anthropic_api_key"}
    values: dict[str, str | None] = {f: None for f in fields.values()}
    sources: dict[str, str] = {}
    for provider, key in (byok or {}).items():
        values[fields[provider]] = key
        sources[provider] = "user"
    for provider, key in (env or {}).items():
        values[fields[provider]] = key
        sources[provider] = "env"
    return ApiKeys(tavily_api_key=None, model_id=model_id, key_sources=sources, **values)


def _model(provider: str, *, lendable: bool):
    return next(m for m in get_all_models() if m.provider == provider and m.operator_lendable is lendable)


def _other_provider_model():
    default = get_default_model()
    return next(m for m in get_all_models() if m.provider != default.provider)


def test_explicit_registered_request_wins_even_without_a_key():
    asked = _model("openai", lendable=False)
    model, key = resolve_model(_keys(byok={"google_genai": "g"}), asked.id)
    assert model.id == asked.id
    assert key is None


def test_explicit_request_uses_the_callers_key():
    asked = _model("openai", lendable=False)
    model, key = resolve_model(_keys(byok={"openai": "o-key"}), asked.id)
    assert (model.id, key) == (asked.id, "o-key")


@pytest.mark.parametrize("bogus", ["retired-model-id", "", None, 42, ["gemini"], {"id": "x"}])
def test_unknown_or_malformed_request_counts_as_no_request(bogus):
    default = get_default_model()
    model, key = resolve_model(_keys(byok={default.provider: "k"}), bogus)
    assert (model.id, key) == (default.id, "k")


def test_default_model_used_when_a_key_runs_it():
    default = get_default_model()
    model, key = resolve_model(_keys(byok={default.provider: "k"}), None)
    assert (model.id, key) == (default.id, "k")


def test_falls_back_to_a_model_the_caller_has_a_byok_key_for():
    other = _other_provider_model()
    model, key = resolve_model(_keys(byok={other.provider: "k"}), None)
    assert model.provider == other.provider
    assert key == "k"
    # The first registry model of that provider.
    assert model.id == next(m.id for m in get_all_models() if m.provider == other.provider)


def test_operator_key_only_counts_for_lendable_models(clerk_env):
    # Hosted deploy (Clerk on), operator holds only a Google key, default is
    # Anthropic: fall back to a lendable Google model on the operator key.
    default = get_default_model()
    assert default.provider != "google_genai"
    model, key = resolve_model(_keys(env={"google_genai": "env-g"}), None)
    assert model.provider == "google_genai"
    assert model.operator_lendable is True
    assert key == "env-g"


def test_operator_key_never_resolves_a_non_lendable_model(clerk_env):
    asked = _model("google_genai", lendable=False)
    model, key = resolve_model(_keys(env={"google_genai": "env-g"}), asked.id)
    assert model.id == asked.id
    assert key is None


def test_self_host_operator_key_runs_any_model():
    # Clerk unconfigured: the env keys belong to the operator, any model is fair.
    asked = _model("google_genai", lendable=False)
    model, key = resolve_model(_keys(env={"google_genai": "env-g"}), asked.id)
    assert (model.id, key) == (asked.id, "env-g")


def test_no_usable_key_falls_to_the_free_trial_model_without_a_key():
    trial = get_free_tier_model()
    assert trial is not None
    model, key = resolve_model(_keys(), None)
    assert model.id == trial.id
    assert key is None


def test_no_usable_key_and_no_trial_falls_to_the_default(monkeypatch):
    monkeypatch.setattr(dependencies, "get_free_tier_model", lambda: None)
    model, key = resolve_model(_keys(), None)
    assert model.id == get_default_model().id
    assert key is None


def test_operator_key_for_a_provider_with_nothing_lendable_is_no_key(clerk_env):
    assert not any(m.provider == "openai" and m.operator_lendable for m in get_all_models())
    model, key = resolve_model(_keys(env={"openai": "env-o"}), None)
    assert model.id == get_free_tier_model().id
    assert key is None


def test_env_default_model_id_is_not_an_explicit_request(monkeypatch):
    # DEFAULT_MODEL_ID is env-defaulted into keys.model_id; with no usable key
    # it must not beat the free-trial fallback.
    trial = get_free_tier_model()
    other = next(m for m in get_all_models() if m.id != trial.id)
    monkeypatch.setenv("DEFAULT_MODEL_ID", other.id)
    model, key = resolve_model(_keys(model_id=other.id), None)
    assert model.id == trial.id
    assert key is None


def test_keys_model_id_steers_the_default_when_keyed():
    # keys.model_id (header/env) is the preferred model, but only as the
    # default: used when a key runs it.
    preferred = _model("openai", lendable=False)
    model, key = resolve_model(_keys(byok={"openai": "o"}, model_id=preferred.id), None)
    assert (model.id, key) == (preferred.id, "o")


def test_returned_model_is_a_registry_entry():
    model, _ = resolve_model(_keys(), None)
    assert get_model(model.id) is model
