"""Tests for GET /models and GET /env-keys endpoints."""

import asyncio

import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

import api.db as db_module
from agents.model_registry import get_all_models, get_free_tier_model
from api.main import app
from api import dependencies
from api.routes import models as models_route
from tests.conftest import clear_operator_keys, set_operator_key


client = TestClient(app)


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    """/env-keys reads today's free-trial usage — keep it off the dev DB."""
    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "models.db")
    monkeypatch.setattr(db_module, "_db", None)
    asyncio.run(db_module.init_db())
    yield
    asyncio.run(db_module.close_db())

_REQUIRED_MODEL_FIELDS = {"id", "provider", "display_name", "max_context", "thinking_capable", "default"}


class TestListModels:
    @pytest.mark.eval_unit
    def test_returns_200(self):
        resp = client.get("/models")
        assert resp.status_code == 200

    @pytest.mark.eval_unit
    def test_response_has_models_key(self):
        resp = client.get("/models")
        assert "models" in resp.json()

    @pytest.mark.eval_unit
    def test_each_model_has_required_fields(self):
        models = client.get("/models").json()["models"]
        for model in models:
            assert _REQUIRED_MODEL_FIELDS <= model.keys()

    @pytest.mark.eval_unit
    def test_exactly_one_default(self):
        models = client.get("/models").json()["models"]
        defaults = [m for m in models if m["default"]]
        assert len(defaults) == 1

    @pytest.mark.eval_unit
    def test_lists_every_registry_model(self):
        models = client.get("/models").json()["models"]
        assert [m["id"] for m in models] == [m.id for m in get_all_models()]


class TestEnvKeys:
    @pytest.mark.eval_unit
    def test_returns_200(self):
        resp = client.get("/env-keys")
        assert resp.status_code == 200

    @pytest.mark.eval_unit
    def test_response_shape(self):
        body = client.get("/env-keys").json()
        assert {"google", "openai", "anthropic", "tavily"} <= body.keys()

    @pytest.mark.eval_unit
    def test_all_values_are_booleans(self):
        # free_trial_model_id (str | None), free_trial_queries (int) and
        # operator_models (list of model ids) are the deliberate exceptions —
        # everything else stays a plain availability flag.
        non_bool_fields = {"free_trial_model_id", "free_trial_queries", "operator_models"}
        body = client.get("/env-keys").json()
        for key, value in body.items():
            if key in non_bool_fields:
                continue
            assert isinstance(value, bool), f"{key!r} is not a bool"

    @pytest.mark.eval_unit
    @patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"})
    def test_reflects_env_var_present(self):
        resp = client.get("/env-keys")
        assert resp.json()["google"] is True

    @pytest.mark.eval_unit
    @patch.dict("os.environ", {}, clear=True)
    def test_reflects_env_var_absent(self):
        resp = client.get("/env-keys")
        assert resp.json()["google"] is False


class TestEnvKeysFreeTrial:
    @pytest.mark.eval_unit
    def test_trial_advertised_while_global_cap_has_room(self, monkeypatch):
        set_operator_key(monkeypatch, get_free_tier_model())
        body = client.get("/env-keys").json()
        assert body["free_trial_model_id"] == models_route.free_trial_model_id()

    @pytest.mark.eval_unit
    def test_trial_hidden_once_global_cap_is_spent(self, monkeypatch):
        # Otherwise the frontend lets anon into a chat that can only error.
        set_operator_key(monkeypatch, get_free_tier_model())
        monkeypatch.setattr(dependencies, "ANON_FREE_DAILY_CAP", 1)
        asyncio.run(db_module.increment_llm_usage(dependencies.FREE_TRIAL_GLOBAL_USAGE_KEY))
        assert client.get("/env-keys").json()["free_trial_model_id"] is None

    @pytest.mark.eval_unit
    def test_trial_hidden_without_operator_key(self, monkeypatch):
        clear_operator_keys(monkeypatch)
        assert client.get("/env-keys").json()["free_trial_model_id"] is None


def _provider_model_ids(provider: str, lendable_only: bool = False) -> list[str]:
    from agents.model_registry import get_all_models

    return [
        m.id for m in get_all_models()
        if m.provider == provider and (m.operator_lendable or not lendable_only)
    ]


@pytest.fixture
def operator_keys(monkeypatch):
    """Operator holds Google + OpenAI keys; Anthropic is unconfigured."""
    monkeypatch.setenv("GOOGLE_API_KEY", "operator-google")
    monkeypatch.setenv("OPENAI_API_KEY", "operator-openai")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


SIGNED_IN = {"X-Clerk-Session-Token": "good-token"}


class TestEnvKeysOperatorModels:
    @pytest.mark.eval_unit
    def test_self_host_covers_every_model_of_a_keyed_provider(self, operator_keys):
        body = client.get("/env-keys").json()
        assert set(body["operator_models"]) == set(
            _provider_model_ids("google_genai") + _provider_model_ids("openai")
        )
        assert body["google"] is True
        assert body["openai"] is True
        assert body["anthropic"] is False

    @pytest.mark.eval_unit
    def test_signed_in_covers_only_lendable_models(self, operator_keys, clerk_on):
        body = client.get("/env-keys", headers=SIGNED_IN).json()
        lendable = _provider_model_ids("google_genai", lendable_only=True)
        assert lendable, "registry should flag at least one Google model lendable"
        assert set(body["operator_models"]) == set(lendable)
        assert body["google"] is True

    @pytest.mark.eval_unit
    def test_provider_with_no_lendable_model_reads_false(self, operator_keys, clerk_on):
        # The operator holds an OpenAI key, but none of its models is lendable:
        # the UI must not tell a signed-in user OpenAI works on the shared key.
        assert not _provider_model_ids("openai", lendable_only=True)
        body = client.get("/env-keys", headers=SIGNED_IN).json()
        assert body["openai"] is False
        assert not set(_provider_model_ids("openai")) & set(body["operator_models"])

    @pytest.mark.eval_unit
    def test_anonymous_covers_nothing_but_keeps_free_trial(
        self, operator_keys, clerk_on, monkeypatch
    ):
        set_operator_key(monkeypatch, get_free_tier_model())
        body = client.get("/env-keys").json()
        assert body["operator_models"] == []
        assert (body["google"], body["openai"], body["anthropic"]) == (False, False, False)
        assert body["free_trial_model_id"] == models_route.free_trial_model_id()

    @pytest.mark.eval_unit
    def test_own_key_is_not_reported_as_operator_covered(self, operator_keys):
        body = client.get("/env-keys", headers={"X-Google-Api-Key": "mine"}).json()
        assert body["google"] is False
        assert not set(_provider_model_ids("google_genai")) & set(body["operator_models"])
