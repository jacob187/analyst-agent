"""Tests for API key resolution logic in api/dependencies.py."""

import pytest
from unittest.mock import patch

from api.dependencies import ApiKeys, get_api_keys, resolve_ws_keys


class TestGetApiKeys:
    @pytest.mark.eval_unit
    async def test_resolves_google_key_from_header(self):
        result = await get_api_keys(x_google_api_key="hdr-key")
        assert result.google_api_key == "hdr-key"

    @pytest.mark.eval_unit
    @patch.dict("os.environ", {"GOOGLE_API_KEY": "env-key"})
    async def test_falls_back_to_env_google(self):
        # Pass None explicitly — FastAPI Header() sentinels are truthy objects
        # that bypass the `or os.getenv()` fallback when called outside HTTP context.
        result = await get_api_keys(x_google_api_key=None)
        assert result.google_api_key == "env-key"

    @pytest.mark.eval_unit
    @patch.dict("os.environ", {"GOOGLE_API_KEY": "env"})
    async def test_header_takes_priority_over_env(self):
        result = await get_api_keys(x_google_api_key="hdr")
        assert result.google_api_key == "hdr"

    @pytest.mark.eval_unit
    async def test_all_providers_resolved(self):
        result = await get_api_keys(
            x_google_api_key="g",
            x_openai_api_key="o",
            x_anthropic_api_key="a",
            x_sec_header="sec",
            x_tavily_api_key="tav",
            x_model_id="gemini-3-flash-preview",
        )
        assert result.google_api_key == "g"
        assert result.openai_api_key == "o"
        assert result.anthropic_api_key == "a"
        assert result.sec_header == "sec"
        assert result.tavily_api_key == "tav"
        assert result.model_id == "gemini-3-flash-preview"

    @pytest.mark.eval_unit
    @patch.dict("os.environ", {}, clear=True)
    async def test_missing_key_is_none(self):
        result = await get_api_keys(x_google_api_key=None)
        assert result.google_api_key is None


class TestResolveWsKeys:
    @pytest.mark.eval_unit
    def test_resolves_from_auth_message(self):
        auth = {
            "google_api_key": "g",
            "openai_api_key": "o",
            "anthropic_api_key": "a",
            "sec_header": "sec",
            "tavily_api_key": "tav",
            "model_id": "gpt-4.1-mini",
        }
        result = resolve_ws_keys(auth)
        assert result.google_api_key == "g"
        assert result.openai_api_key == "o"
        assert result.anthropic_api_key == "a"
        assert result.sec_header == "sec"
        assert result.tavily_api_key == "tav"
        assert result.model_id == "gpt-4.1-mini"

    @pytest.mark.eval_unit
    @patch.dict("os.environ", {"GOOGLE_API_KEY": "env-g"})
    def test_falls_back_to_env(self):
        result = resolve_ws_keys({})
        assert result.google_api_key == "env-g"

    @pytest.mark.eval_unit
    @patch.dict("os.environ", {"GOOGLE_API_KEY": "env-g"})
    def test_auth_takes_priority_over_env(self):
        result = resolve_ws_keys({"google_api_key": "auth-g"})
        assert result.google_api_key == "auth-g"


class TestApiKeysGetProviderKey:
    @pytest.mark.eval_unit
    def test_google_genai_provider(self):
        keys = ApiKeys(
            google_api_key="k",
            openai_api_key=None,
            anthropic_api_key=None,
            sec_header=None,
            tavily_api_key=None,
            model_id=None,
        )
        assert keys.get_provider_key("google_genai") == "k"

    @pytest.mark.eval_unit
    def test_openai_provider(self):
        keys = ApiKeys(
            google_api_key=None,
            openai_api_key="k",
            anthropic_api_key=None,
            sec_header=None,
            tavily_api_key=None,
            model_id=None,
        )
        assert keys.get_provider_key("openai") == "k"

    @pytest.mark.eval_unit
    def test_anthropic_provider(self):
        keys = ApiKeys(
            google_api_key=None,
            openai_api_key=None,
            anthropic_api_key="k",
            sec_header=None,
            tavily_api_key=None,
            model_id=None,
        )
        assert keys.get_provider_key("anthropic") == "k"

    @pytest.mark.eval_unit
    def test_unknown_provider_returns_none(self):
        keys = ApiKeys(
            google_api_key="g",
            openai_api_key=None,
            anthropic_api_key=None,
            sec_header=None,
            tavily_api_key=None,
            model_id=None,
        )
        assert keys.get_provider_key("unknown") is None
