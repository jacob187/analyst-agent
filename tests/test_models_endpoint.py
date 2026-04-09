"""Tests for GET /models and GET /env-keys endpoints."""

import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)

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
        assert "gemini-3-flash-preview" in ids


class TestEnvKeys:
    @pytest.mark.eval_unit
    def test_returns_200(self):
        resp = client.get("/env-keys")
        assert resp.status_code == 200

    @pytest.mark.eval_unit
    def test_response_shape(self):
        body = client.get("/env-keys").json()
        assert {"google", "openai", "anthropic", "sec_header", "tavily"} <= body.keys()

    @pytest.mark.eval_unit
    def test_all_values_are_booleans(self):
        body = client.get("/env-keys").json()
        for key, value in body.items():
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
