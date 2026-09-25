"""Tests for GET /models and GET /env-keys endpoints."""

import asyncio

import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

import api.db as db_module
from api.main import app
from api.routes import models as models_route


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
    def test_known_model_present(self):
        models = client.get("/models").json()["models"]
        ids = {m["id"] for m in models}
        assert "gemini-3.6-flash" in ids


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
        # free_trial_model_id (str | None) and free_trial_queries (int) are the
        # deliberate exceptions — everything else stays a plain availability flag.
        non_bool_fields = {"free_trial_model_id", "free_trial_queries"}
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
        monkeypatch.setenv("GOOGLE_API_KEY", "operator-key")
        body = client.get("/env-keys").json()
        assert body["free_trial_model_id"] == models_route.free_trial_model_id()

    @pytest.mark.eval_unit
    def test_trial_hidden_once_global_cap_is_spent(self, monkeypatch):
        # Otherwise the frontend lets anon into a chat that can only error.
        monkeypatch.setenv("GOOGLE_API_KEY", "operator-key")
        monkeypatch.setattr(models_route, "ANON_FREE_DAILY_CAP", 1)
        asyncio.run(db_module.increment_llm_usage(models_route.FREE_TRIAL_GLOBAL_USAGE_KEY))
        assert client.get("/env-keys").json()["free_trial_model_id"] is None

    @pytest.mark.eval_unit
    def test_trial_hidden_without_operator_key(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        assert client.get("/env-keys").json()["free_trial_model_id"] is None
