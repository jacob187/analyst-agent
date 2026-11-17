"""api.clerk_auth: JWKS key handling against a local fake JWKS endpoint.

The real PyJWKClient fetch, JSON parsing and key-set validation run; only
urllib's opener is replaced (tests/clerk_fakes.py), so there is no network.
"""

import json
import threading
import time
from urllib.error import URLError

import pytest

from api import clerk_auth
from api.clerk_auth import ClerkVerifierUnavailable, verify_clerk_token
from tests.clerk_fakes import FakeJwksEndpoint, SigningKey

ISSUER = "https://test.clerk.test"
SUB = "user_2abcDEF123"

pytestmark = pytest.mark.eval_unit


@pytest.fixture(scope="module")
def key():
    return SigningKey("kid-current")


@pytest.fixture(scope="module")
def rotated_key():
    return SigningKey("kid-rotated")


@pytest.fixture(autouse=True)
def clerk_issuer(monkeypatch):
    monkeypatch.setenv("CLERK_SECRET_KEY", "sk_test_xxx")
    monkeypatch.setenv("CLERK_ISSUER_URL", ISSUER)
    monkeypatch.delenv("CLERK_JWKS_URL", raising=False)
    clerk_auth._reset_cache_for_tests()
    yield
    clerk_auth._reset_cache_for_tests()


@pytest.fixture
def jwks(key, monkeypatch):
    return FakeJwksEndpoint(key).install(monkeypatch)


def _token(signing_key: SigningKey, **kw) -> str:
    return signing_key.token(sub=SUB, issuer=ISSUER, **kw)


def _in_thread(fn):
    """Run fn() in a thread; return (thread, result dict with value/error/elapsed)."""
    out: dict = {}

    def run():
        start = time.monotonic()
        try:
            out["value"] = fn()
        except BaseException as e:  # noqa: BLE001 - surfaced to the test
            out["error"] = e
        out["elapsed"] = time.monotonic() - start

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t, out


def _wait_for_fetches(jwks: FakeJwksEndpoint, n: int) -> None:
    deadline = time.monotonic() + 5
    while jwks.calls < n:
        assert time.monotonic() < deadline, f"expected {n} JWKS fetches, saw {jwks.calls}"
        time.sleep(0.01)


class TestValidAndInvalidTokens:
    def test_valid_token_verifies_with_one_fetch(self, jwks, key):
        assert verify_clerk_token(_token(key)) == SUB
        assert verify_clerk_token(_token(key)) == SUB
        assert jwks.calls == 1

    def test_garbage_token_is_anonymous_without_a_fetch(self, jwks):
        assert verify_clerk_token("not-a-jwt") is None
        assert jwks.calls == 0

    def test_expired_token_is_anonymous(self, jwks, key):
        assert verify_clerk_token(_token(key, exp_delta=-60)) is None

    def test_unknown_kid_after_a_successful_refresh_is_anonymous(self, jwks, rotated_key):
        # The JWKS answered and doesn't have this key: a bad token, not an outage.
        assert verify_clerk_token(_token(rotated_key)) is None

    def test_bad_signature_is_anonymous(self, jwks, key):
        forged = SigningKey(key.kid)  # same kid, different private key
        assert verify_clerk_token(_token(forged)) is None

    def test_rotated_key_is_picked_up_by_a_refresh(self, jwks, key, rotated_key, monkeypatch):
        assert verify_clerk_token(_token(key)) == SUB
        jwks.serve_keys(key, rotated_key)
        monkeypatch.setattr(clerk_auth, "_JWKS_UNKNOWN_KID_COOLDOWN_SECONDS", 0)
        assert verify_clerk_token(_token(rotated_key)) == SUB
        assert jwks.calls == 2

    def test_unknown_kids_cannot_force_a_fetch_per_request(self, jwks, key, rotated_key):
        assert verify_clerk_token(_token(key)) == SUB
        for _ in range(5):
            assert verify_clerk_token(_token(rotated_key)) is None
        assert jwks.calls == 1


class TestJwksOutage:
    """#6: a JWKS outage must not drop keys already known."""

    def test_known_key_keeps_verifying_after_the_ttl_when_jwks_is_down(
        self, jwks, key, monkeypatch
    ):
        assert verify_clerk_token(_token(key)) == SUB
        jwks.fail()
        monkeypatch.setattr(clerk_auth, "_JWKS_TTL_SECONDS", 1e-6)  # key set now stale
        for _ in range(3):
            assert verify_clerk_token(_token(key)) == SUB
        # One failed refresh, then no retry storm while it is down.
        assert jwks.calls == 2

    def test_unknown_kid_during_an_outage_is_unavailable(
        self, jwks, key, rotated_key, monkeypatch, caplog
    ):
        assert verify_clerk_token(_token(key)) == SUB
        jwks.fail()
        monkeypatch.setattr(clerk_auth, "_JWKS_TTL_SECONDS", 1e-6)
        with caplog.at_level("ERROR", logger="api.clerk_auth"):
            with pytest.raises(ClerkVerifierUnavailable):
                verify_clerk_token(_token(rotated_key))
            # The known key still verifies during the same outage.
            assert verify_clerk_token(_token(key)) == SUB
        assert any(r.levelname == "ERROR" for r in caplog.records)

    def test_cold_start_outage_is_unavailable(self, jwks, key):
        jwks.fail()
        with pytest.raises(ClerkVerifierUnavailable):
            verify_clerk_token(_token(key))

    def test_failed_fetch_is_not_retried_by_every_request(self, jwks, key):
        jwks.fail()
        for _ in range(5):
            with pytest.raises(ClerkVerifierUnavailable):
                verify_clerk_token(_token(key))
        assert jwks.calls == 1

    def test_recovers_after_the_retry_delay(self, jwks, key, monkeypatch):
        jwks.fail()
        with pytest.raises(ClerkVerifierUnavailable):
            verify_clerk_token(_token(key))
        jwks.serve_keys(key)
        monkeypatch.setattr(clerk_auth, "_JWKS_RETRY_AFTER_FAILURE_SECONDS", 0)
        assert verify_clerk_token(_token(key)) == SUB

    def test_refresh_drops_keys_the_jwks_no_longer_lists(
        self, jwks, key, rotated_key, monkeypatch
    ):
        assert verify_clerk_token(_token(key)) == SUB
        jwks.serve_keys(rotated_key)
        monkeypatch.setattr(clerk_auth, "_JWKS_TTL_SECONDS", 1e-6)
        assert verify_clerk_token(_token(key)) is None


class TestStalledJwks:
    """#25: a stalled JWKS host must not queue requests behind it."""

    def test_fetch_uses_a_short_timeout(self, jwks, key):
        verify_clerk_token(_token(key))
        assert jwks.timeouts and all(t is not None and t <= 5 for t in jwks.timeouts)

    def test_known_key_is_served_while_a_refresh_stalls(self, jwks, key, monkeypatch):
        assert verify_clerk_token(_token(key)) == SUB
        monkeypatch.setattr(clerk_auth, "_JWKS_TTL_SECONDS", 1e-6)
        gate = jwks.hold(timeout=3)
        stalled, _ = _in_thread(lambda: verify_clerk_token(_token(key)))
        _wait_for_fetches(jwks, 2)  # the refresh is in flight, stalled

        start = time.monotonic()
        assert verify_clerk_token(_token(key)) == SUB
        assert time.monotonic() - start < 1.0
        assert jwks.calls == 2  # no second fetch behind the stalled one

        gate.set()
        stalled.join(5)

    def test_cold_start_requests_share_one_fetch(self, jwks, key):
        gate = jwks.hold(timeout=3)
        workers = [_in_thread(lambda: verify_clerk_token(_token(key))) for _ in range(5)]
        _wait_for_fetches(jwks, 1)
        time.sleep(0.05)
        gate.set()
        for t, out in workers:
            t.join(5)
            assert out.get("value") == SUB, out
        assert jwks.calls == 1

    def test_waiters_give_up_after_the_fetch_timeout(self, jwks, key, monkeypatch):
        monkeypatch.setattr(clerk_auth, "_JWKS_FETCH_TIMEOUT_SECONDS", 0.2)
        gate = jwks.hold(timeout=3)
        fetcher, _ = _in_thread(lambda: verify_clerk_token(_token(key)))
        _wait_for_fetches(jwks, 1)
        start = time.monotonic()
        with pytest.raises(ClerkVerifierUnavailable):
            verify_clerk_token(_token(key))
        assert time.monotonic() - start < 1.0
        gate.set()
        fetcher.join(5)


class TestVerifierFaults:
    """#26: a JWKS that isn't a usable key set is a verifier fault (503),
    not an invalid token."""

    @pytest.mark.parametrize("body", [
        json.dumps({}).encode(),
        json.dumps({"keys": []}).encode(),
        json.dumps([1, 2]).encode(),
        json.dumps({"keys": [{"kty": "RSA", "use": "enc", "kid": "x", "n": "AQAB", "e": "AQAB"}]}).encode(),
        b"<!doctype html><title>Not Found</title>",
        b"\xff\xfe not utf-8",
    ], ids=["empty-object", "no-keys", "array", "no-signing-keys", "html", "binary"])
    def test_unusable_jwks_is_unavailable(self, jwks, key, body, caplog):
        jwks.serve_raw(body)
        with caplog.at_level("ERROR", logger="api.clerk_auth"):
            with pytest.raises(ClerkVerifierUnavailable):
                verify_clerk_token(_token(key))
        assert any(r.levelname == "ERROR" for r in caplog.records)

    def test_schemeless_issuer_is_unavailable(self, jwks, key, monkeypatch):
        monkeypatch.setenv("CLERK_ISSUER_URL", "test.clerk.test")
        with pytest.raises(ClerkVerifierUnavailable):
            verify_clerk_token(key.token(sub=SUB, issuer="test.clerk.test"))
        assert jwks.calls == 0

    def test_connection_error_is_unavailable(self, jwks, key):
        jwks.fail(URLError("timed out"))
        with pytest.raises(ClerkVerifierUnavailable):
            verify_clerk_token(_token(key))

    def test_socket_error_mid_read_is_unavailable(self, jwks, key):
        jwks.fail(ConnectionResetError("reset by peer"))
        with pytest.raises(ClerkVerifierUnavailable):
            verify_clerk_token(_token(key))
