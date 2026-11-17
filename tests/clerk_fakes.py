"""Test fakes for Clerk verification: RSA signing keys and an in-memory JWKS
endpoint.

``FakeJwksEndpoint.install(monkeypatch)`` replaces the urllib opener that
PyJWKClient fetches with, so the real fetch, JSON parsing and key-set
validation run against the body set here, with no network. ``calls`` counts
fetches and ``timeouts`` records the timeout each one was given.
"""

from __future__ import annotations

import base64
import io
import json
import threading
import time
from typing import Any
from urllib.error import URLError

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa


def _b64u(n: int) -> str:
    raw = n.to_bytes((n.bit_length() + 7) // 8, "big")
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


class SigningKey:
    """An RS256 key pair with a key id: signs tokens, publishes its JWK."""

    def __init__(self, kid: str) -> None:
        self.kid = kid
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        self.private_pem = key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        numbers = key.public_key().public_numbers()
        self.jwk = {
            "kty": "RSA", "kid": kid, "use": "sig", "alg": "RS256",
            "n": _b64u(numbers.n), "e": _b64u(numbers.e),
        }

    def token(self, *, sub: str, issuer: str, exp_delta: int = 300) -> str:
        now = int(time.time())
        return jwt.encode(
            {"sub": sub, "iss": issuer, "exp": now + exp_delta, "iat": now},
            self.private_pem, algorithm="RS256", headers={"kid": self.kid},
        )


class FakeJwksEndpoint:
    """The JWKS URL's responses. Serves ``{"keys": [...]}`` for the given
    keys until told to serve something else, fail, or hold requests."""

    def __init__(self, *keys: SigningKey) -> None:
        self.calls = 0
        self.timeouts: list[float | None] = []
        self._body = b""
        self._error: BaseException | None = None
        self._gate: threading.Event | None = None
        self._gate_timeout = 0.0
        self._lock = threading.Lock()
        self.serve_keys(*keys)

    def install(self, monkeypatch) -> "FakeJwksEndpoint":
        monkeypatch.setattr(
            "jwt.jwks_client.urllib.request.build_opener", lambda *handlers: self
        )
        return self

    def serve_keys(self, *keys: SigningKey) -> None:
        self.serve_json({"keys": [k.jwk for k in keys]})

    def serve_json(self, payload: Any) -> None:
        self.serve_raw(json.dumps(payload).encode())

    def serve_raw(self, body: bytes) -> None:
        self._body, self._error = body, None

    def fail(self, error: BaseException | None = None) -> None:
        """Every fetch raises ``error`` (default: connection refused)."""
        self._error = error or URLError("connection refused")

    def hold(self, timeout: float = 3.0) -> threading.Event:
        """Stall every fetch until the returned event is set (or ``timeout``
        passes): a JWKS host that accepts the connection but doesn't answer."""
        self._gate, self._gate_timeout = threading.Event(), timeout
        return self._gate

    # urllib OpenerDirector surface used by PyJWKClient.fetch_data
    def open(self, request, timeout=None):
        with self._lock:
            self.calls += 1
            self.timeouts.append(timeout)
        if self._gate is not None:
            self._gate.wait(self._gate_timeout)
        if self._error is not None:
            raise self._error
        return io.BytesIO(self._body)
