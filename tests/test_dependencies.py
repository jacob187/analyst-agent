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
    free_trial_key,
    free_trial_model_id,
    get_api_keys,
    resolve_ws_identity,
    resolve_ws_keys,
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
            x_model_id="gemini-3.6-flash",
        )
        assert result.google_api_key == "g"
        assert result.openai_api_key == "o"
        assert result.anthropic_api_key == "a"
        assert result.tavily_api_key == "tav"
        assert result.model_id == "gemini-3.6-flash"

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

    @pytest.mark.eval_unit
    async def test_anonymous_uuid_dropped_without_token(self, clerk_env):
        # Progressive auth: a legacy UUID is not a Clerk account and can't be
        # verified, so with Clerk enabled and no token it is dropped to anonymous
        # (None) — trusting it would let anyone claim any UUID's data.
        anon = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
        result = await get_api_keys(x_user_id=anon, x_clerk_session_token=None)
        assert result.user_id is None


class TestTokenIsIdentityRest:
    """With Clerk enabled, a verified token alone identifies the caller."""

    @pytest.mark.eval_unit
    async def test_token_alone_yields_user_id(self, clerk_env, rsa_keypair):
        priv, _ = rsa_keypair
        result = await get_api_keys(x_user_id=None, x_clerk_session_token=_make_token(priv))
        assert result.user_id == TEST_SUB

    @pytest.mark.eval_unit
    async def test_token_alone_verified_user_inherits_env_key(
        self, clerk_env, rsa_keypair, monkeypatch
    ):
        monkeypatch.setenv("GOOGLE_API_KEY", "env-key")
        priv, _ = rsa_keypair
        result = await get_api_keys(
            x_google_api_key=None, x_openai_api_key=None,
            x_anthropic_api_key=None, x_tavily_api_key=None,
            x_user_id=None, x_clerk_session_token=_make_token(priv),
        )
        assert result.google_api_key == "env-key"
        assert result.is_operator_paid("google_genai") is True

    @pytest.mark.eval_unit
    async def test_invalid_token_alone_degrades_to_anonymous(self, clerk_env, caplog):
        # Session tokens are short-lived; an expired one must not break public
        # endpoints. Failure is logged (never the token itself).
        with caplog.at_level("INFO", logger="api.dependencies"):
            result = await get_api_keys(x_user_id=None, x_clerk_session_token="not-a-jwt")
        assert result.user_id is None
        assert any("failed verification" in r.message for r in caplog.records)
        assert all("not-a-jwt" not in r.getMessage() for r in caplog.records)

    @pytest.mark.eval_unit
    async def test_expired_token_alone_degrades_to_anonymous(self, clerk_env, rsa_keypair):
        priv, _ = rsa_keypair
        token = _make_token(priv, exp_delta=-60)
        result = await get_api_keys(x_user_id=None, x_clerk_session_token=token)
        assert result.user_id is None

    @pytest.mark.eval_unit
    async def test_invalid_token_with_user_id_rejected(self, clerk_env):
        with pytest.raises(HTTPException) as exc:
            await get_api_keys(x_user_id=TEST_SUB, x_clerk_session_token="not-a-jwt")
        assert exc.value.status_code == 401

    @pytest.mark.eval_unit
    async def test_invalid_token_alone_gets_no_env_key(self, clerk_env, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "env-key")
        result = await get_api_keys(
            x_google_api_key=None, x_openai_api_key=None,
            x_anthropic_api_key=None, x_tavily_api_key=None,
            x_user_id=None, x_clerk_session_token="not-a-jwt",
        )
        assert result.google_api_key is None

    @pytest.mark.eval_unit
    async def test_legacy_uuid_user_id_conflicts_with_token(self, clerk_env, rsa_keypair):
        priv, _ = rsa_keypair
        with pytest.raises(HTTPException) as exc:
            await get_api_keys(
                x_user_id="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                x_clerk_session_token=_make_token(priv),
            )
        assert exc.value.status_code == 401

    @pytest.mark.eval_unit
    async def test_auth_disabled_ignores_token_and_keeps_user_id(
        self, clerk_env, monkeypatch
    ):
        monkeypatch.setenv("DISABLE_AUTH", "true")
        result = await get_api_keys(x_user_id=TEST_SUB, x_clerk_session_token="whatever")
        assert result.user_id == TEST_SUB
        result = await get_api_keys(x_user_id=None, x_clerk_session_token="whatever")
        assert result.user_id is None


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """Throwaway SQLite file with the schema initialised (real DB, no mocks)."""
    import asyncio

    import api.db as db_module

    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "deps.db")
    monkeypatch.setattr(db_module, "_db", None)
    asyncio.run(db_module.init_db())
    yield db_module
    asyncio.run(db_module.close_db())


class TestTokenIdentityEndpoint:
    """End-to-end through TestClient: `GET /sessions/{id}/messages` scopes the
    session lookup to the user id derived from the Clerk token."""

    @pytest.fixture
    def seeded(self, temp_db):
        import asyncio

        async def _seed():
            session_id = await temp_db.get_or_create_session("AAPL", TEST_SUB)
            await temp_db.save_message(session_id, "user", "hello from the token user")
            return session_id

        return asyncio.run(_seed())

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient

        from api.main import app

        return TestClient(app)

    @pytest.mark.eval_unit
    def test_token_only_returns_users_messages(self, clerk_env, rsa_keypair, seeded, client):
        priv, _ = rsa_keypair
        resp = client.get(
            f"/sessions/{seeded}/messages",
            params={"ticker": "AAPL"},
            headers={"X-Clerk-Session-Token": _make_token(priv)},
        )
        assert resp.status_code == 200
        assert [m["content"] for m in resp.json()["messages"]] == ["hello from the token user"]

    @pytest.mark.eval_unit
    def test_token_with_matching_user_id_also_works(self, clerk_env, rsa_keypair, seeded, client):
        priv, _ = rsa_keypair
        resp = client.get(
            f"/sessions/{seeded}/messages",
            params={"ticker": "AAPL"},
            headers={"X-Clerk-Session-Token": _make_token(priv), "X-User-Id": TEST_SUB},
        )
        assert resp.status_code == 200

    @pytest.mark.eval_unit
    def test_user_id_disagreeing_with_token_is_401(self, clerk_env, rsa_keypair, seeded, client):
        priv, _ = rsa_keypair
        resp = client.get(
            f"/sessions/{seeded}/messages",
            params={"ticker": "AAPL"},
            headers={"X-Clerk-Session-Token": _make_token(priv), "X-User-Id": "user_someoneElse"},
        )
        assert resp.status_code == 401

    @pytest.mark.eval_unit
    def test_other_users_token_cannot_read_session(self, clerk_env, rsa_keypair, seeded, client):
        priv, _ = rsa_keypair
        resp = client.get(
            f"/sessions/{seeded}/messages",
            params={"ticker": "AAPL"},
            headers={"X-Clerk-Session-Token": _make_token(priv, sub="user_intruder")},
        )
        assert resp.status_code == 404

    @pytest.mark.eval_unit
    def test_invalid_token_alone_per_user_endpoint_is_422(self, clerk_env, seeded, client):
        resp = client.get(
            f"/sessions/{seeded}/messages",
            params={"ticker": "AAPL"},
            headers={"X-Clerk-Session-Token": "not-a-jwt"},
        )
        assert resp.status_code == 422

    @pytest.mark.eval_unit
    def test_expired_token_alone_public_endpoint_is_200(self, clerk_env, rsa_keypair, client):
        priv, _ = rsa_keypair
        resp = client.get(
            "/models",
            headers={"X-Clerk-Session-Token": _make_token(priv, exp_delta=-60)},
        )
        assert resp.status_code == 200

    @pytest.mark.eval_unit
    def test_invalid_token_with_user_id_is_401(self, clerk_env, seeded, client):
        resp = client.get(
            f"/sessions/{seeded}/messages",
            params={"ticker": "AAPL"},
            headers={"X-Clerk-Session-Token": "not-a-jwt", "X-User-Id": TEST_SUB},
        )
        assert resp.status_code == 401

    @pytest.mark.eval_unit
    def test_no_token_is_anonymous_422(self, clerk_env, seeded, client):
        resp = client.get(f"/sessions/{seeded}/messages", params={"ticker": "AAPL"})
        assert resp.status_code == 422

    @pytest.mark.eval_unit
    def test_user_id_without_token_is_401(self, clerk_env, seeded, client):
        resp = client.get(
            f"/sessions/{seeded}/messages",
            params={"ticker": "AAPL"},
            headers={"X-User-Id": TEST_SUB},
        )
        assert resp.status_code == 401

    @pytest.mark.eval_unit
    def test_auth_disabled_user_id_header_still_works(self, no_clerk_env, seeded, client):
        resp = client.get(
            f"/sessions/{seeded}/messages",
            params={"ticker": "AAPL"},
            headers={"X-User-Id": TEST_SUB},
        )
        assert resp.status_code == 200
        assert len(resp.json()["messages"]) == 1


class TestEnvKeyGate:
    """With Clerk enabled, the operator's env keys are lent only to verified
    signed-in users — never to anonymous callers. This is the cost boundary:
    anonymous visitors may use LLM features only with their own BYOK keys.
    """

    @pytest.mark.eval_unit
    async def test_anon_does_not_inherit_env_key(self, clerk_env, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "env-key")
        result = await get_api_keys(
            x_google_api_key=None, x_openai_api_key=None,
            x_anthropic_api_key=None, x_tavily_api_key=None,
            x_user_id=None, x_clerk_session_token=None,
        )
        assert result.google_api_key is None
        assert result.is_operator_paid("google_genai") is False

    @pytest.mark.eval_unit
    async def test_anon_uuid_does_not_inherit_env_key(self, clerk_env, monkeypatch):
        # A legacy UUID is dropped to anonymous under Clerk, so it neither
        # becomes a trusted identity nor unlocks operator env keys.
        monkeypatch.setenv("GOOGLE_API_KEY", "env-key")
        anon = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
        result = await get_api_keys(
            x_google_api_key=None, x_openai_api_key=None,
            x_anthropic_api_key=None, x_tavily_api_key=None,
            x_user_id=anon, x_clerk_session_token=None,
        )
        assert result.user_id is None
        assert result.google_api_key is None

    @pytest.mark.eval_unit
    async def test_anon_byok_key_still_resolves(self, clerk_env, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "env-key")
        result = await get_api_keys(
            x_google_api_key="byok", x_openai_api_key=None,
            x_anthropic_api_key=None, x_tavily_api_key=None,
            x_user_id=None, x_clerk_session_token=None,
        )
        assert result.google_api_key == "byok"
        assert result.is_operator_paid("google_genai") is False

    @pytest.mark.eval_unit
    async def test_verified_user_inherits_env_key(self, clerk_env, rsa_keypair, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "env-key")
        priv, _ = rsa_keypair
        token = _make_token(priv)
        result = await get_api_keys(
            x_google_api_key=None, x_openai_api_key=None,
            x_anthropic_api_key=None, x_tavily_api_key=None,
            x_user_id=TEST_SUB, x_clerk_session_token=token,
        )
        assert result.google_api_key == "env-key"
        assert result.is_operator_paid("google_genai") is True

    @pytest.mark.eval_unit
    async def test_self_host_anon_inherits_env_key(self, no_clerk_env, monkeypatch):
        # Clerk unconfigured (local/self-host): the single operator is trusted
        # with env keys even without a user_id.
        monkeypatch.setenv("GOOGLE_API_KEY", "env-key")
        result = await get_api_keys(
            x_google_api_key=None, x_openai_api_key=None,
            x_anthropic_api_key=None, x_tavily_api_key=None,
            x_user_id=None, x_clerk_session_token=None,
        )
        assert result.google_api_key == "env-key"

    @pytest.mark.eval_unit
    def test_ws_anon_does_not_inherit_env_key(self, clerk_env, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "env-key")
        result = resolve_ws_keys({})
        assert result.google_api_key is None

    @pytest.mark.eval_unit
    async def test_env_keys_endpoint_hides_provider_from_anon(self, clerk_env, monkeypatch):
        # /env-keys must not advertise the operator key to anonymous callers,
        # or the UI renders chat/filings affordances the backend then refuses.
        from api.routes.models import env_keys
        monkeypatch.setenv("GOOGLE_API_KEY", "env-key")
        keys = await get_api_keys(
            x_google_api_key=None, x_openai_api_key=None,
            x_anthropic_api_key=None, x_tavily_api_key=None,
            x_user_id=None, x_clerk_session_token=None,
        )
        body = await env_keys(keys)
        assert body["google"] is False

    @pytest.mark.eval_unit
    async def test_env_keys_endpoint_shows_provider_to_verified(self, clerk_env, rsa_keypair, monkeypatch):
        from api.routes.models import env_keys
        monkeypatch.setenv("GOOGLE_API_KEY", "env-key")
        priv, _ = rsa_keypair
        token = _make_token(priv)
        keys = await get_api_keys(
            x_google_api_key=None, x_openai_api_key=None,
            x_anthropic_api_key=None, x_tavily_api_key=None,
            x_user_id=TEST_SUB, x_clerk_session_token=token,
        )
        body = await env_keys(keys)
        assert body["google"] is True


class TestClerkWsVerification:
    @pytest.mark.eval_unit
    def test_valid_token_matching_user_ok(self, clerk_env, rsa_keypair):
        priv, _ = rsa_keypair
        token = _make_token(priv)
        user_id, reason = resolve_ws_identity(
            {"user_id": TEST_SUB, "clerk_session_token": token}
        )
        assert user_id == TEST_SUB
        assert reason is None

    @pytest.mark.eval_unit
    def test_token_alone_yields_user_id(self, clerk_env, rsa_keypair):
        # The token's verified subject is the identity; user_id is optional.
        priv, _ = rsa_keypair
        token = _make_token(priv)
        user_id, reason = resolve_ws_identity({"clerk_session_token": token})
        assert user_id == TEST_SUB
        assert reason is None

    @pytest.mark.eval_unit
    def test_mismatched_user_ws_rejected(self, clerk_env, rsa_keypair):
        priv, _ = rsa_keypair
        token = _make_token(priv, sub="user_other")
        user_id, reason = resolve_ws_identity(
            {"user_id": TEST_SUB, "clerk_session_token": token}
        )
        assert user_id is None
        assert "match" in (reason or "").lower()

    @pytest.mark.eval_unit
    def test_invalid_token_ws_alone_is_anonymous(self, clerk_env):
        assert resolve_ws_identity({"clerk_session_token": "not-a-jwt"}) == (None, None)

    @pytest.mark.eval_unit
    def test_invalid_token_ws_with_user_id_rejected(self, clerk_env):
        user_id, reason = resolve_ws_identity(
            {"user_id": TEST_SUB, "clerk_session_token": "not-a-jwt"}
        )
        assert user_id is None
        assert "invalid" in (reason or "").lower()

    @pytest.mark.eval_unit
    def test_non_string_token_ws_treated_as_absent(self, clerk_env):
        user_id, reason = resolve_ws_identity({"clerk_session_token": 123})
        assert user_id is None
        assert reason is None

    @pytest.mark.eval_unit
    def test_missing_token_ws_rejected(self, clerk_env):
        user_id, reason = resolve_ws_identity({"user_id": TEST_SUB})
        assert user_id is None
        assert "missing" in (reason or "").lower()

    @pytest.mark.eval_unit
    def test_no_credentials_ws_is_anonymous(self, clerk_env):
        assert resolve_ws_identity({}) == (None, None)

    @pytest.mark.eval_unit
    def test_ws_clerk_unconfigured_trusts_user_id(self, no_clerk_env):
        user_id, reason = resolve_ws_identity({"user_id": TEST_SUB})
        assert user_id == TEST_SUB
        assert reason is None

    @pytest.mark.eval_unit
    def test_ws_disable_auth_trusts_user_id(self, clerk_env, monkeypatch):
        monkeypatch.setenv("DISABLE_AUTH", "true")
        user_id, reason = resolve_ws_identity({"user_id": TEST_SUB})
        assert user_id == TEST_SUB
        assert reason is None

    @pytest.mark.eval_unit
    def test_ws_anonymous_uuid_ok_without_token(self, clerk_env):
        # A non-Clerk id connects as anonymous (no token needed) and is
        # dropped to None so nothing persists under an unverifiable identity.
        anon = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
        assert resolve_ws_identity({"user_id": anon}) == (None, None)

    @pytest.mark.eval_unit
    def test_ws_legacy_uuid_identity_dropped(self, clerk_env):
        # resolve_ws_keys mirrors REST: a legacy UUID is unverifiable under
        # Clerk, so it resolves to anonymous.
        anon = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
        result = resolve_ws_keys({"user_id": anon})
        assert result.user_id is None

    @pytest.mark.eval_unit
    def test_ws_keys_token_only_gets_env_key(self, clerk_env, rsa_keypair, monkeypatch):
        # Env-key lending keys off the verified id, not the user_id field.
        monkeypatch.setenv("GOOGLE_API_KEY", "env-key")
        priv, _ = rsa_keypair
        auth = {"clerk_session_token": _make_token(priv)}
        user_id, _reason = resolve_ws_identity(auth)
        result = resolve_ws_keys(auth, user_id=user_id)
        assert result.user_id == TEST_SUB
        assert result.google_api_key == "env-key"
        assert result.is_operator_paid("google_genai") is True

    @pytest.mark.eval_unit
    def test_ws_keys_token_only_resolves_identity_itself(self, clerk_env, rsa_keypair):
        priv, _ = rsa_keypair
        result = resolve_ws_keys({"clerk_session_token": _make_token(priv)})
        assert result.user_id == TEST_SUB


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


class TestIsOperatorPaid:
    """Provenance tracking on ApiKeys — the daily LLM budget only applies
    when the provider key came from server env, not when the user supplied
    their own via header / WS auth message.

    These tests pass `None` explicitly for header args — FastAPI Header()
    defaults are truthy sentinels when get_api_keys() is called outside
    an HTTP request context.
    """

    @pytest.mark.eval_unit
    async def test_env_only_key_is_operator_paid(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "env-key")
        keys = await get_api_keys(
            x_google_api_key=None, x_openai_api_key=None,
            x_anthropic_api_key=None, x_tavily_api_key=None,
        )
        assert keys.is_operator_paid("google_genai") is True

    @pytest.mark.eval_unit
    async def test_header_key_is_not_operator_paid(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "env-key")
        keys = await get_api_keys(
            x_google_api_key="user-key", x_openai_api_key=None,
            x_anthropic_api_key=None, x_tavily_api_key=None,
        )
        # Header wins; provenance says "user" so operator isn't paying.
        assert keys.is_operator_paid("google_genai") is False
        assert keys.google_api_key == "user-key"

    @pytest.mark.eval_unit
    async def test_no_key_is_not_operator_paid(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        keys = await get_api_keys(
            x_google_api_key=None, x_openai_api_key=None,
            x_anthropic_api_key=None, x_tavily_api_key=None,
        )
        assert keys.is_operator_paid("google_genai") is False
        assert keys.google_api_key is None

    @pytest.mark.eval_unit
    async def test_per_provider_independent(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "env-google")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        keys = await get_api_keys(
            x_google_api_key=None, x_openai_api_key="user-openai",
            x_anthropic_api_key=None, x_tavily_api_key=None,
        )
        assert keys.is_operator_paid("google_genai") is True
        assert keys.is_operator_paid("openai") is False

    @pytest.mark.eval_unit
    def test_ws_auth_message_keys_are_user_provenance(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "env-key")
        keys = resolve_ws_keys({"google_api_key": "ws-user-key"})
        assert keys.is_operator_paid("google_genai") is False

    @pytest.mark.eval_unit
    def test_ws_env_fallback_is_operator_paid(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "env-key")
        keys = resolve_ws_keys({})
        assert keys.is_operator_paid("google_genai") is True

    @pytest.mark.eval_unit
    def test_unknown_provider_returns_false(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "env-key")
        keys = resolve_ws_keys({})
        assert keys.is_operator_paid("not_a_real_provider") is False


class TestFreeTrial:
    """Anonymous visitors may spend the operator's key for exactly one light
    model (the free trial), independent of the Clerk-gated env-key resolver.
    """

    @pytest.mark.eval_unit
    def test_defaults_to_registry_free_tier_model(self, monkeypatch):
        from agents.model_registry import get_free_tier_model
        monkeypatch.delenv("ANON_FREE_MODEL_ID", raising=False)
        assert free_trial_model_id() == get_free_tier_model().id

    @pytest.mark.eval_unit
    def test_respects_env_override(self, monkeypatch):
        monkeypatch.setenv("ANON_FREE_MODEL_ID", "claude-sonnet-5")
        assert free_trial_model_id() == "claude-sonnet-5"

    @pytest.mark.eval_unit
    def test_trial_disabled_when_no_model_flagged(self, monkeypatch):
        # Unflagging free_tier turns the trial off; it must not fall through to
        # lending the operator's key for the default model.
        import agents.model_registry as registry
        monkeypatch.delenv("ANON_FREE_MODEL_ID", raising=False)
        monkeypatch.setenv("GOOGLE_API_KEY", "env-key")
        monkeypatch.setattr(registry, "_FREE_TIER_MODEL", None)
        assert free_trial_model_id() is None
        assert free_trial_key() is None

    @pytest.mark.eval_unit
    def test_no_key_when_operator_has_not_configured_provider(self, monkeypatch):
        monkeypatch.delenv("ANON_FREE_MODEL_ID", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        assert free_trial_key() is None

    @pytest.mark.eval_unit
    def test_key_resolves_when_operator_has_configured_provider(self, monkeypatch):
        monkeypatch.delenv("ANON_FREE_MODEL_ID", raising=False)
        monkeypatch.setenv("GOOGLE_API_KEY", "env-key")
        assert free_trial_key() == "env-key"

    @pytest.mark.eval_unit
    async def test_env_keys_endpoint_reports_free_trial_to_anon(self, clerk_env, monkeypatch):
        # Even though clerk_env gates the ordinary provider booleans away from
        # anon (google=False), the free-trial model id is still surfaced — the
        # one deliberate exception to that gate.
        from api.routes.models import env_keys
        monkeypatch.delenv("ANON_FREE_MODEL_ID", raising=False)
        monkeypatch.setenv("GOOGLE_API_KEY", "env-key")
        keys = await get_api_keys(
            x_google_api_key=None, x_openai_api_key=None,
            x_anthropic_api_key=None, x_tavily_api_key=None,
            x_user_id=None, x_clerk_session_token=None,
        )
        body = await env_keys(keys)
        assert body["google"] is False
        assert body["free_trial_model_id"] == free_trial_model_id()

    @pytest.mark.eval_unit
    async def test_env_keys_endpoint_omits_free_trial_without_operator_key(self, clerk_env, monkeypatch):
        from api.routes.models import env_keys
        monkeypatch.delenv("ANON_FREE_MODEL_ID", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        keys = await get_api_keys(
            x_google_api_key=None, x_openai_api_key=None,
            x_anthropic_api_key=None, x_tavily_api_key=None,
            x_user_id=None, x_clerk_session_token=None,
        )
        body = await env_keys(keys)
        assert body["free_trial_model_id"] is None
