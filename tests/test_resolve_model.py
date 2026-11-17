"""`api.dependencies.resolve_model`: the model a request runs on and the key
that may run it, shared by chat and the filings endpoints.

Pure logic over real `ApiKeys` and the real registry — no mocks.
"""

import pytest

from agents.model_registry import get_all_models, get_default_model, get_free_tier_model, get_model
from api import dependencies
from api.dependencies import _PROVIDER_KEY_FIELDS, ApiKeys, resolve_model
from tests.conftest import clear_operator_keys, model_where, other_model
from tests.test_dependencies import clerk_env, rsa_keypair  # noqa: F401  (fixtures)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    clear_operator_keys(monkeypatch)
    monkeypatch.delenv("DEFAULT_MODEL_ID", raising=False)


def _keys(*, byok: dict[str, str] | None = None, env: dict[str, str] | None = None,
          model_id: str | None = None) -> ApiKeys:
    """ApiKeys with per-provider keys marked as caller-supplied or operator env."""
    fields = _PROVIDER_KEY_FIELDS
    values: dict[str, str | None] = {f: None for f in fields.values()}
    sources: dict[str, str] = {}
    for provider, key in (byok or {}).items():
        values[fields[provider]] = key
        sources[provider] = "user"
    for provider, key in (env or {}).items():
        values[fields[provider]] = key
        sources[provider] = "env"
    return ApiKeys(tavily_api_key=None, model_id=model_id, key_sources=sources, **values)


def _other_provider_model():
    default = get_default_model()
    match = next((m for m in get_all_models() if m.provider != default.provider), None)
    if match is None:
        pytest.skip("registry has a single provider")
    return match


def _another_provider(than: str) -> str:
    """Any provider with a key field other than `than`."""
    return next(p for p in _PROVIDER_KEY_FIELDS if p != than)


def _lendable_non_default_provider_model():
    """A lendable model from a provider other than the default's."""
    default = get_default_model()
    match = next(
        (m for m in get_all_models() if m.operator_lendable and m.provider != default.provider), None
    )
    if match is None:
        pytest.skip("no lendable model outside the default's provider")
    return match


def test_explicit_registered_request_wins_even_without_a_key():
    asked = model_where(operator_lendable=False)
    model, key = resolve_model(_keys(byok={_another_provider(asked.provider): "k"}), asked.id)
    assert model.id == asked.id
    assert key is None


def test_explicit_request_uses_the_callers_key():
    asked = model_where(operator_lendable=False)
    model, key = resolve_model(_keys(byok={asked.provider: "o-key"}), asked.id)
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


def _price(model):
    if model.input_price_per_mtok is None or model.output_price_per_mtok is None:
        return None
    return model.input_price_per_mtok + model.output_price_per_mtok


def test_falls_back_to_a_model_the_caller_has_a_byok_key_for():
    other = _other_provider_model()
    model, key = resolve_model(_keys(byok={other.provider: "k"}), None)
    assert model.provider == other.provider
    assert key == "k"


def test_fallback_is_the_cheapest_runnable_model_not_the_first_in_file():
    # Find a provider (not the default's, whose key would run the default)
    # whose first model in models.json isn't its cheapest priced one.
    default = get_default_model()
    for provider in {m.provider for m in get_all_models()} - {default.provider}:
        mine = [m for m in get_all_models() if m.provider == provider]
        priced = [m for m in mine if _price(m) is not None]
        if priced and _price(mine[0]) != min(_price(m) for m in priced):
            break
    else:
        pytest.skip("every provider lists its cheapest model first")
    model, _ = resolve_model(_keys(byok={provider: "k"}), None)
    assert _price(model) == min(_price(m) for m in priced)


def _fake(model_id, provider, inp=None, out=None):
    from agents.model_registry import ModelDef

    return ModelDef(
        id=model_id, provider=provider, display_name=model_id, max_context=1000,
        thinking_capable=False, default=False,
        input_price_per_mtok=inp, output_price_per_mtok=out,
    )


@pytest.fixture
def fake_registry(monkeypatch):
    """Swap the fallback's model list for fakes on a provider the default
    doesn't use, so the default never runs and step 3 decides."""
    provider = _another_provider(get_default_model().provider)

    def install(*specs):
        models = [_fake(f"fake-model-{i}", provider, *spec) for i, spec in enumerate(specs)]
        monkeypatch.setattr(dependencies, "get_all_models", lambda: models)
        return provider, models

    return install


def test_unpriced_models_rank_after_priced_ones(fake_registry):
    provider, models = fake_registry((None, None), (5.0, 20.0), (2.0, None))
    model, _ = resolve_model(_keys(byok={provider: "k"}), None)
    assert model is models[1]


def test_price_ties_keep_file_order(fake_registry):
    provider, models = fake_registry((3.0, 1.0), (1.0, 3.0), (0.5, 3.5))
    model, _ = resolve_model(_keys(byok={provider: "k"}), None)
    assert model is models[0]


def test_all_unpriced_falls_back_to_file_order(fake_registry):
    provider, models = fake_registry((None, None), (None, None))
    model, _ = resolve_model(_keys(byok={provider: "k"}), None)
    assert model is models[0]


def test_operator_key_only_counts_for_lendable_models(clerk_env):
    # Hosted deploy (Clerk on), the operator holds a key only for a provider
    # other than the default's: fall back to a lendable model on that key.
    provider = _lendable_non_default_provider_model().provider
    model, key = resolve_model(_keys(env={provider: "env-k"}), None)
    assert model.provider == provider
    assert model.operator_lendable is True
    assert key == "env-k"


def test_operator_key_never_resolves_a_non_lendable_model(clerk_env):
    asked = model_where(operator_lendable=False)
    model, key = resolve_model(_keys(env={asked.provider: "env-k"}), asked.id)
    assert model.id == asked.id
    assert key is None


def test_self_host_operator_key_runs_any_model():
    # Clerk unconfigured: the env keys belong to the operator, any model is fair.
    asked = model_where(operator_lendable=False)
    model, key = resolve_model(_keys(env={asked.provider: "env-k"}), asked.id)
    assert (model.id, key) == (asked.id, "env-k")


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
    lendable_providers = {m.provider for m in get_all_models() if m.operator_lendable}
    provider = next((p for p in _PROVIDER_KEY_FIELDS if p not in lendable_providers), None)
    if provider is None:
        pytest.skip("every provider has a lendable model")
    model, key = resolve_model(_keys(env={provider: "env-k"}), None)
    assert model.id == get_free_tier_model().id
    assert key is None


def test_env_default_model_id_is_not_an_explicit_request(monkeypatch):
    # DEFAULT_MODEL_ID is env-defaulted into keys.model_id; with no usable key
    # it must not beat the free-trial fallback.
    trial = get_free_tier_model()
    other = other_model(trial)
    monkeypatch.setenv("DEFAULT_MODEL_ID", other.id)
    model, key = resolve_model(_keys(model_id=other.id), None)
    assert model.id == trial.id
    assert key is None


def test_keys_model_id_steers_the_default_when_keyed():
    # keys.model_id (header/env) is the preferred model, but only as the
    # default: used when a key runs it.
    preferred = other_model(get_default_model(), operator_lendable=False)
    model, key = resolve_model(_keys(byok={preferred.provider: "k"}, model_id=preferred.id), None)
    assert (model.id, key) == (preferred.id, "k")


def test_returned_model_is_a_registry_entry():
    model, _ = resolve_model(_keys(), None)
    assert get_model(model.id) is model
