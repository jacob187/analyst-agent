"""Tests for API key resolution logic in api/dependencies.py."""

import time
from unittest.mock import patch

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import HTTPException

from api import clerk_auth
from api.dependencies import (
    ApiKeys,
    _validate_user_id,
    get_api_keys,
    resolve_ws_keys,
    verify_ws_identity,
)


@pytest.fixture(autouse=True)
def _disable_clerk_by_default(monkeypatch):
    """Default to Clerk-disabled. Tests that need Clerk re-enable via `clerk_env`.

    The project's .env may set CLERK_SECRET_KEY for the dev server; without
    this fixture, every legacy test would see Clerk enabled and 401 on
    missing tokens.
    """
    monkeypatch.delenv("CLERK_SECRET_KEY", raising=False)
    monkeypatch.delenv("DISABLE_AUTH", raising=False)
    from api import dependencies
    dependencies._clerk_unconfigured_warned = False


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
            x_tavily_api_key="tav",
            x_model_id="gemini-3-flash-preview",
        )
        assert result.google_api_key == "g"
        assert result.openai_api_key == "o"
        assert result.anthropic_api_key == "a"
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
            "tavily_api_key": "tav",
            "model_id": "gpt-4.1-mini",
        }
        result = resolve_ws_keys(auth)
        assert result.google_api_key == "g"
        assert result.openai_api_key == "o"
        assert result.anthropic_api_key == "a"
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
            tavily_api_key=None,
            model_id=None,
        )
        assert keys.get_provider_key("unknown") is None


class TestUserIdValidation:
    @pytest.mark.eval_unit
    def test_valid_uuid(self):
        assert _validate_user_id("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa") == "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"

    @pytest.mark.eval_unit
    def test_none_returns_none(self):
        assert _validate_user_id(None) is None

    @pytest.mark.eval_unit
    def test_empty_string_returns_none(self):
        assert _validate_user_id("") is None

    @pytest.mark.eval_unit
    def test_invalid_format_returns_none(self):
        assert _validate_user_id("not-a-uuid") is None

    @pytest.mark.eval_unit
    def test_uppercase_uuid_rejected(self):
        """UUIDs must be lowercase hex."""
        assert _validate_user_id("AAAAAAAA-AAAA-AAAA-AAAA-AAAAAAAAAAAA") is None


class TestUserIdInGetApiKeys:
    @pytest.mark.eval_unit
    async def test_resolves_user_id_from_header(self):
        result = await get_api_keys(x_user_id="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
        assert result.user_id == "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"

    @pytest.mark.eval_unit
    async def test_invalid_user_id_becomes_none(self):
        result = await get_api_keys(x_user_id="bad-value")
        assert result.user_id is None

    @pytest.mark.eval_unit
    async def test_missing_user_id_is_none(self):
        result = await get_api_keys()
        assert result.user_id is None


class TestUserIdInWsKeys:
    @pytest.mark.eval_unit
    def test_resolves_user_id_from_auth(self):
        result = resolve_ws_keys({"user_id": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"})
        assert result.user_id == "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"

    @pytest.mark.eval_unit
    def test_invalid_user_id_in_auth(self):
        result = resolve_ws_keys({"user_id": "garbage"})
        assert result.user_id is None


class TestRequireUserId:
    @pytest.mark.eval_unit
    def test_returns_id_when_present(self):
        keys = ApiKeys(
            google_api_key=None, openai_api_key=None, anthropic_api_key=None,
            tavily_api_key=None, model_id=None,
            user_id="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        )
        assert keys.require_user_id() == "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"

    @pytest.mark.eval_unit
    def test_raises_when_missing(self):
        keys = ApiKeys(
            google_api_key=None, openai_api_key=None, anthropic_api_key=None,
            tavily_api_key=None, model_id=None,
            user_id=None,
        )
        with pytest.raises(ValueError):
            keys.require_user_id()


# ---------------------------------------------------------------------------
# Clerk verification — uses a test RSA keypair; mocks only the JWKS HTTP fetch.
# ---------------------------------------------------------------------------

TEST_ISSUER = "https://test.clerk.test"
TEST_KID = "test-key-id"
TEST_SUB = "user_2abcDEF123"


@pytest.fixture(scope="module")
def rsa_keypair():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_numbers = key.public_key().public_numbers()
    # Build a JWK manually so PyJWKClient can consume it.
    import base64

    def b64u(n: int) -> str:
        b = n.to_bytes((n.bit_length() + 7) // 8, "big")
        return base64.urlsafe_b64encode(b).rstrip(b"=").decode()

    jwk = {
        "kty": "RSA",
        "kid": TEST_KID,
        "use": "sig",
        "alg": "RS256",
        "n": b64u(public_numbers.n),
        "e": b64u(public_numbers.e),
    }
    return private_pem, jwk


def _make_token(private_pem: bytes, *, sub: str = TEST_SUB, exp_delta: int = 300,
                issuer: str = TEST_ISSUER) -> str:
    now = int(time.time())
    return jwt.encode(
        {"sub": sub, "iss": issuer, "exp": now + exp_delta, "iat": now},
        private_pem,
        algorithm="RS256",
        headers={"kid": TEST_KID},
    )


@pytest.fixture
def clerk_env(rsa_keypair, monkeypatch):
    """Enable Clerk for the test and stub the JWKS fetch."""
    _, jwk = rsa_keypair
    monkeypatch.setenv("CLERK_SECRET_KEY", "sk_test_xxx")
    monkeypatch.setenv("CLERK_ISSUER_URL", TEST_ISSUER)
    monkeypatch.delenv("DISABLE_AUTH", raising=False)
    clerk_auth._reset_cache_for_tests()

    class _FakeJWK:
        def __init__(self, key):
            self.key = key

    def _fake_get_signing_key_from_jwt(self, token):
        from jwt.algorithms import RSAAlgorithm
        return _FakeJWK(RSAAlgorithm.from_jwk(jwk))

    monkeypatch.setattr(
        "jwt.PyJWKClient.get_signing_key_from_jwt",
        _fake_get_signing_key_from_jwt,
    )
    # Reset the one-time-warning flag so tests are independent.
    from api import dependencies
    dependencies._clerk_unconfigured_warned = False
    yield
    clerk_auth._reset_cache_for_tests()


@pytest.fixture
def no_clerk_env(monkeypatch):
    monkeypatch.delenv("CLERK_SECRET_KEY", raising=False)
    monkeypatch.delenv("DISABLE_AUTH", raising=False)
    from api import dependencies
    dependencies._clerk_unconfigured_warned = False


class TestClerkRestVerification:
    @pytest.mark.eval_unit
    async def test_valid_token_matching_user_accepted(self, clerk_env, rsa_keypair):
        priv, _ = rsa_keypair
        token = _make_token(priv)
        result = await get_api_keys(x_user_id=TEST_SUB, x_clerk_session_token=token)
        assert result.user_id == TEST_SUB

    @pytest.mark.eval_unit
    async def test_valid_token_mismatched_user_rejected(self, clerk_env, rsa_keypair):
        priv, _ = rsa_keypair
        token = _make_token(priv, sub="user_someoneElse")
        with pytest.raises(HTTPException) as exc:
            await get_api_keys(x_user_id=TEST_SUB, x_clerk_session_token=token)
        assert exc.value.status_code == 401

    @pytest.mark.eval_unit
    async def test_expired_token_rejected(self, clerk_env, rsa_keypair):
        priv, _ = rsa_keypair
        token = _make_token(priv, exp_delta=-60)
        with pytest.raises(HTTPException) as exc:
            await get_api_keys(x_user_id=TEST_SUB, x_clerk_session_token=token)
        assert exc.value.status_code == 401

    @pytest.mark.eval_unit
    async def test_wrong_issuer_rejected(self, clerk_env, rsa_keypair):
        priv, _ = rsa_keypair
        token = _make_token(priv, issuer="https://attacker.example")
        with pytest.raises(HTTPException) as exc:
            await get_api_keys(x_user_id=TEST_SUB, x_clerk_session_token=token)
        assert exc.value.status_code == 401

    @pytest.mark.eval_unit
    async def test_garbage_token_rejected(self, clerk_env):
        with pytest.raises(HTTPException) as exc:
            await get_api_keys(x_user_id=TEST_SUB, x_clerk_session_token="not-a-jwt")
        assert exc.value.status_code == 401

    @pytest.mark.eval_unit
    async def test_clerk_enabled_no_token_rejected(self, clerk_env):
        with pytest.raises(HTTPException) as exc:
            await get_api_keys(x_user_id=TEST_SUB, x_clerk_session_token=None)
        assert exc.value.status_code == 401

    @pytest.mark.eval_unit
    async def test_clerk_unconfigured_passes_through(self, no_clerk_env):
        # No token, no Clerk env — dev mode accepts the user_id as-is.
        result = await get_api_keys(x_user_id=TEST_SUB, x_clerk_session_token=None)
        assert result.user_id == TEST_SUB

    @pytest.mark.eval_unit
    async def test_disable_auth_bypasses_verification(self, clerk_env, monkeypatch):
        monkeypatch.setenv("DISABLE_AUTH", "true")
        # No token, but DISABLE_AUTH wins.
        result = await get_api_keys(x_user_id=TEST_SUB, x_clerk_session_token=None)
        assert result.user_id == TEST_SUB

    @pytest.mark.eval_unit
    async def test_no_user_id_passes_through(self, clerk_env):
        # When there's no user_id we don't try to verify; endpoint-level
        # require_user_id is responsible for rejecting.
        result = await get_api_keys(x_user_id=None, x_clerk_session_token=None)
        assert result.user_id is None


class TestClerkWsVerification:
    @pytest.mark.eval_unit
    def test_valid_token_matching_user_ok(self, clerk_env, rsa_keypair):
        priv, _ = rsa_keypair
        token = _make_token(priv)
        ok, reason = verify_ws_identity(TEST_SUB, {"clerk_session_token": token})
        assert ok is True
        assert reason is None

    @pytest.mark.eval_unit
    def test_mismatched_user_ws_rejected(self, clerk_env, rsa_keypair):
        priv, _ = rsa_keypair
        token = _make_token(priv, sub="user_other")
        ok, reason = verify_ws_identity(TEST_SUB, {"clerk_session_token": token})
        assert ok is False
        assert "match" in (reason or "").lower()

    @pytest.mark.eval_unit
    def test_missing_token_ws_rejected(self, clerk_env):
        ok, reason = verify_ws_identity(TEST_SUB, {})
        assert ok is False
        assert "missing" in (reason or "").lower()

    @pytest.mark.eval_unit
    def test_ws_clerk_unconfigured_ok(self, no_clerk_env):
        ok, reason = verify_ws_identity(TEST_SUB, {})
        assert ok is True
        assert reason is None

    @pytest.mark.eval_unit
    def test_ws_disable_auth_ok(self, clerk_env, monkeypatch):
        monkeypatch.setenv("DISABLE_AUTH", "true")
        ok, reason = verify_ws_identity(TEST_SUB, {})
        assert ok is True
        assert reason is None


class TestUserIdRegexClerkFormat:
    @pytest.mark.eval_unit
    def test_clerk_format_accepted(self):
        assert _validate_user_id("user_2abcDEF123xyz") == "user_2abcDEF123xyz"

    @pytest.mark.eval_unit
    def test_legacy_uuid_still_accepted(self):
        uuid = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
        assert _validate_user_id(uuid) == uuid

    @pytest.mark.eval_unit
    def test_user_prefix_without_suffix_rejected(self):
        assert _validate_user_id("user_") is None

    @pytest.mark.eval_unit
    def test_user_prefix_with_hyphen_rejected(self):
        # Clerk IDs are alphanumeric only after the underscore.
        assert _validate_user_id("user_abc-def") is None
